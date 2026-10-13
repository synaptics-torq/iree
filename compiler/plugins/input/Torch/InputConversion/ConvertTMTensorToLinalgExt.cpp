// Copyright 2022 The IREE Authors
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <mlir/IR/BuiltinAttributes.h>
#include <cstdint>
#include <numeric>

#include "compiler/plugins/input/Torch/InputConversion/Passes.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtDialect.h"
#include "iree/compiler/Dialect/LinalgExt/IR/LinalgExtOps.h"
#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "torch-mlir-dialects/Dialect/TMTensor/IR/TMTensorOps.h"

namespace mlir::iree_compiler::TorchInput {

#define GEN_PASS_DEF_CONVERTTMTENSORTOLINALGEXTPASS
#include "compiler/plugins/input/Torch/InputConversion/Passes.h.inc"

namespace {

template <typename SrcOpTy, typename TargetOpTy>
struct TMTensorOpConversion : OpRewritePattern<SrcOpTy> {
  using OpRewritePattern<SrcOpTy>::OpRewritePattern;
  LogicalResult matchAndRewrite(SrcOpTy srcOp,
                                PatternRewriter &rewriter) const override {
    OperationState state(srcOp->getLoc(), TargetOpTy::getOperationName(),
                         srcOp->getOperands(), srcOp->getResultTypes(),
                         srcOp->getAttrs(), srcOp->getSuccessors());
    for (Region &srcRegion : srcOp->getRegions()) {
      Region *targetRegion = state.addRegion();
      rewriter.inlineRegionBefore(srcRegion, *targetRegion,
                                  targetRegion->begin());
    }
    Operation *targetOp = rewriter.create(state);
    rewriter.replaceOp(srcOp, targetOp->getResults());
    return success();
  }
};

// Rewrites the `tm_tensor.sort` + leading-`tensor.extract_slice` pattern that
// torch-mlir's DecomposeAtenTopkOp emits for `aten.topk` (sort the whole dim,
// keep the first k) into a single `tm_tensor.topk`. The sort reference
// lowering is O(N^2); the topk scalar lowering is O(N*K), which for detection
// heads (topk of 8400 anchors down to 300) is a ~28x reduction in host work.
// It also drops the i64 index tensors of the sort: tm_tensor.topk infers
// i32 indices from the input position, the index width the NPU data path
// supports.
struct TopkFromSortPattern
    : public OpRewritePattern<mlir::torch::TMTensor::SortOp> {
  // Higher benefit than TMTensorOpConversion<SortOp> so the topk fold is
  // attempted before the plain sort conversion.
  TopkFromSortPattern(MLIRContext *context)
      : OpRewritePattern<mlir::torch::TMTensor::SortOp>(context, 2) {}

  // The sort comparison emitted by createTMTensorSortOp is `>=`/`<=` (the
  // stable-ordering form).  Topk's region wants the strict form (`>`/`<`);
  // the topk scalar implementation applies first-occurrence tie-breaking
  // itself.  Maps the sort predicate to the topk predicate; fails on
  // predicates the sort creation path does not emit.
  static std::optional<arith::CmpFPredicate>
  mapFloatPredicate(arith::CmpFPredicate p) {
    switch (p) {
    case arith::CmpFPredicate::OGE:
    case arith::CmpFPredicate::OGT:
      return arith::CmpFPredicate::OGT;
    case arith::CmpFPredicate::OLE:
    case arith::CmpFPredicate::OLT:
      return arith::CmpFPredicate::OLT;
    default:
      return std::nullopt;
    }
  }

  static std::optional<arith::CmpIPredicate>
  mapIntPredicate(arith::CmpIPredicate p) {
    switch (p) {
    case arith::CmpIPredicate::sge:
    case arith::CmpIPredicate::sgt:
      return arith::CmpIPredicate::sgt;
    case arith::CmpIPredicate::sle:
    case arith::CmpIPredicate::slt:
      return arith::CmpIPredicate::slt;
    case arith::CmpIPredicate::uge:
    case arith::CmpIPredicate::ugt:
      return arith::CmpIPredicate::ugt;
    case arith::CmpIPredicate::ule:
    case arith::CmpIPredicate::ult:
      return arith::CmpIPredicate::ult;
    default:
      return std::nullopt;
    }
  }

  // The sort's index input (when present) must be the identity iota over the
  // sort dimension: `linalg.generic outs(%i) { %ix = linalg.index d; %c =
  // arith.index_cast %ix to <elem>; yield %c }`.  tm_tensor.topk's inferred
  // indices are exactly this, so it can replace the sort without the input.
  static bool isIotaOverDim(Value v, int64_t dim) {
    auto gen = v.getDefiningOp<linalg::GenericOp>();
    if (!gen || gen.getNumDpsInputs() != 0 || gen.getNumDpsInits() != 1)
      return false;
    for (AffineMap m : gen.getIndexingMapsArray())
      if (!m.isIdentity())
        return false;
    for (utils::IteratorType it : gen.getIteratorTypesArray())
      if (it != utils::IteratorType::parallel)
        return false;
    Block &body = gen.getRegion().front();
    auto *indexOp = &body.front();
    if (!isa<linalg::IndexOp>(indexOp) ||
        static_cast<int64_t>(cast<linalg::IndexOp>(indexOp).getDim()) != dim)
      return false;
    auto *yieldOp = body.getTerminator();
    auto *mid = indexOp->getNextNode();
    if (mid == yieldOp) {
      // Iota without an index_cast: the element type must be index-compatible
      // (not produced by the torch sort path, defensive bail).
      return false;
    }
    if (!isa<arith::IndexCastOp>(mid) || mid->getNextNode() != yieldOp ||
        mid->getNumOperands() != 1 || mid->getNumResults() != 1 ||
        mid->getOperand(0) != indexOp->getResult(0) ||
        yieldOp->getNumOperands() != 1 ||
        yieldOp->getOperand(0) != mid->getResult(0))
      return false;
    return cast<RankedTensorType>(gen.getDpsInitOperand(0)->get().getType())
               .getElementType() ==
           mid->getResult(0).getType();
  }

  LogicalResult matchAndRewrite(mlir::torch::TMTensor::SortOp sort,
                                PatternRewriter &rewriter) const override {
    Location loc = sort.getLoc();
    int64_t dim = sort.getDimension();

    // tm_tensor.sort is in-place DPS: the operands to sort are carried in the
    // outputs slot (the creation path passes empty inputs).
    ValueRange sortOuts = sort.getOutputs();
    if (!sort.getInputs().empty() || sortOuts.empty())
      return rewriter.notifyMatchFailure(sort, "unexpected sort operand layout");
    Value values = sortOuts[0];
    auto valuesTy = dyn_cast<RankedTensorType>(values.getType());
    if (!valuesTy || !valuesTy.hasStaticShape() ||
        (int64_t)valuesTy.getRank() <= dim)
      return rewriter.notifyMatchFailure(sort, "values not statically ranked");

    // The index operand, when present, must be the identity iota.
    if (sortOuts.size() > 1 && !isIotaOverDim(sortOuts[1], dim))
      return rewriter.notifyMatchFailure(sort, "sort index input is not the "
                                                "identity iota over the sort dim");

    // Every use of a sort result must be the same leading slice [0, k) along
    // `dim` (or the result is dead).
    unsigned numResults = sort.getNumResults();
    int64_t k = ShapedType::kDynamic;
    SmallVector<tensor::ExtractSliceOp> slices(numResults, nullptr);
    for (unsigned r = 0; r < numResults; ++r) {
      Value result = sort.getResult(r);
      bool sawSlice = false;
      for (OpOperand &use : result.getUses()) {
        auto slice = dyn_cast<tensor::ExtractSliceOp>(use.getOwner());
        if (!slice || slice.getSource() != result) {
          return rewriter.notifyMatchFailure(sort, "sort result has a use "
                                                   "other than the k-slice");
        }
        auto offsets = slice.getMixedOffsets();
        auto sizes = slice.getMixedSizes();
        auto strides = slice.getMixedStrides();
        if (offsets.size() != valuesTy.getRank())
          return failure();
        int64_t sz;
        for (int64_t i = 0; i < valuesTy.getRank(); ++i) {
          bool isDim = (i == dim);
          int64_t full = isDim ? ShapedType::kDynamic : valuesTy.getDimSize(i);
          auto offVal = getConstantIntValue(offsets[i]);
          auto stVal = getConstantIntValue(strides[i]);
          auto szVal = getConstantIntValue(sizes[i]);
          if (!offVal || !stVal || !szVal || stVal.value() != 1 ||
              offVal.value() != 0)
            return rewriter.notifyMatchFailure(sort, "slice offsets/strides not "
                                                     "the leading identity range");
          sz = szVal.value();
          if (isDim ? (sz <= 0) : (sz != full))
            return rewriter.notifyMatchFailure(sort, "slice not [0, k) along the "
                                                     "sort dim over full extents");
          if (isDim)
            k = sz;
        }
        if (sawSlice && slices[r] != slice)
          return rewriter.notifyMatchFailure(sort, "differing slices of a "
                                                   "result");
        slices[r] = slice;
        sawSlice = true;
      }
      if (!sawSlice && k == ShapedType::kDynamic)
        continue; // dead result, no slice info yet
    }
    if (k == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(sort, "no live k-slice to fold");
    if (k > valuesTy.getDimSize(dim))
      return rewriter.notifyMatchFailure(sort, "k exceeds the input extent");

    // The region must be the single value comparison the sort creation path
    // emits (plus its yield).
    Block &sortBlock = sort.getRegion().front();
    auto *yieldOp = sortBlock.getTerminator();
    if (std::distance(sortBlock.begin(), yieldOp->getIterator()) != 1 ||
        yieldOp->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(sort, "sort region not a single "
                                               "comparison");
    Operation *cmp = &sortBlock.front();
    Type elemTy = valuesTy.getElementType();
    std::optional<arith::CmpFPredicate> fpred;
    std::optional<arith::CmpIPredicate> ipred;
    if (auto fcmp = dyn_cast<arith::CmpFOp>(cmp)) {
      fpred = mapFloatPredicate(fcmp.getPredicate());
      if (!fpred || fcmp.getLhs() != sortBlock.getArgument(0) ||
          fcmp.getRhs() != sortBlock.getArgument(1) ||
          yieldOp->getOperand(0) != fcmp->getResult(0))
        return rewriter.notifyMatchFailure(sort, "unsupported float compare");
    } else if (auto icmp = dyn_cast<arith::CmpIOp>(cmp)) {
      ipred = mapIntPredicate(icmp.getPredicate());
      if (!ipred || icmp.getLhs() != sortBlock.getArgument(0) ||
          icmp.getRhs() != sortBlock.getArgument(1) ||
          yieldOp->getOperand(0) != icmp->getResult(0))
        return rewriter.notifyMatchFailure(sort, "unsupported int compare");
    } else {
      return rewriter.notifyMatchFailure(sort, "sort region op is not a compare");
    }

    // Build the topk with the rank-preserving shape (values dim -> k); a
    // rank-reducing slice (dropping size-1 dims, as the torch importer emits
    // for leading unit dims) is restored with a collapse_shape after.
    SmallVector<int64_t> topkShape(valuesTy.getShape());
    topkShape[dim] = k;
    auto topkValuesTy = RankedTensorType::get(topkShape, elemTy);
    auto topkIndicesTy = RankedTensorType::get(topkShape, rewriter.getI32Type());

    // The topk scalar implementation walks the output buffer on every
    // candidate, so the buffer's initial contents matter: a values element
    // that no candidate can beat (top-k) / that every candidate beats
    // (min-k), and an index that loses every first-occurrence tie (i32 max).
    bool topK;
    TypedAttr fillValue;
    if (auto fty = dyn_cast<FloatType>(elemTy)) {
      topK = fpred.has_value() && fpred.value() == arith::CmpFPredicate::OGT;
      fillValue = rewriter.getFloatAttr(
          fty, APFloat::getInf(fty.getFloatSemantics(), /*Negative=*/topK));
    } else if (auto ity = dyn_cast<IntegerType>(elemTy)) {
      topK = ipred.has_value() &&
             (ipred.value() == arith::CmpIPredicate::sgt ||
              ipred.value() == arith::CmpIPredicate::ugt);
      fillValue = rewriter.getIntegerAttr(
          ity, topK ? APInt::getMinValue(ity.getWidth())
                    : APInt::getMaxValue(ity.getWidth()));
    } else {
      return rewriter.notifyMatchFailure(sort, "unsupported topk element type");
    }
    Value fillValueConst =
        rewriter.create<arith::ConstantOp>(loc, fillValue);
    Value fillIndexConst = rewriter.create<arith::ConstantOp>(
        loc, rewriter.getIntegerAttr(rewriter.getI32Type(), APInt::getMaxValue(32)));

    Value emptyValues =
        rewriter.create<tensor::EmptyOp>(loc, topkShape, elemTy);
    Value filledValues =
        rewriter.create<linalg::FillOp>(loc, fillValueConst, emptyValues).getResult(0);
    Value emptyIndices =
        rewriter.create<tensor::EmptyOp>(loc, topkShape, rewriter.getI32Type());
    Value filledIndices =
        rewriter.create<linalg::FillOp>(loc, fillIndexConst, emptyIndices).getResult(0);

    // Create the tm topk, then fill its region (create-then-fill, exactly as
    // createTMTensorTopkOp does for kthvalue: ops must be created in live IR,
    // not in a detached OperationState region).  The
    // TMTensorOpConversion<TopkOp> registered below converts it to an
    // iree_linalg_ext.topk (inlining the region, then the yield conversion
    // finishes the region).
    auto topk = mlir::torch::TMTensor::TopkOp::create(
        rewriter, loc, SmallVector<Type>{topkValuesTy, topkIndicesTy},
        ValueRange{values}, ValueRange{filledValues, filledIndices},
        rewriter.getI64IntegerAttr(dim));
    {
      Region *region = &topk.getRegion();
      Block *block = rewriter.createBlock(region);
      block->addArgument(elemTy, loc);
      block->addArgument(elemTy, loc);
      Operation *newCmp;
      if (fpred) {
        newCmp = arith::CmpFOp::create(rewriter, loc, fpred.value(),
                                       block->getArgument(0), block->getArgument(1));
      } else {
        newCmp = arith::CmpIOp::create(rewriter, loc, ipred.value(),
                                       block->getArgument(0), block->getArgument(1));
      }
      mlir::torch::TMTensor::YieldOp::create(rewriter, loc, newCmp->getResult(0));
    }

    // createBlock() above moved the insertion point into the topk's region
    // block; restore it to a live location (after the just-created topk)
    // before creating the collapse/extsi ops below, or they would land
    // inside the topk region (after its yield) and corrupt it.
    rewriter.setInsertionPointAfter(topk);

    // Rewire the slices' uses to the topk results and drop the old ops.  The
    // topk infers i32 indices (the NPU/host width the topk lowering uses);
    // where the sliced sort result was wider (torch's i64), restore that type
    // at the boundary so downstream i64-typed ops (and the function signature)
    // are unchanged.  The restored i64 stays on the host via the existing
    // i64-component routing; the topk itself already emits i32 internally.
    for (unsigned r = 0; r < numResults; ++r) {
      if (!slices[r])
        continue;
      auto sliceTy = dyn_cast<RankedTensorType>(slices[r].getResult().getType());
      auto topkTy = cast<RankedTensorType>(topk->getResult(r).getType());
      if (!sliceTy || !topkTy) {
        return rewriter.notifyMatchFailure(sort, "slice result not ranked");
      }
      Value replacement = topk->getResult(r);
      if (sliceTy.getShape() != ArrayRef<int64_t>(topkShape)) {
        // Rank-reducing slice: derive the reassociation that collapses the
        // dropped size-1 dims (leading/middle dropped dims attach to the next
        // kept group, trailing ones to the previous).
        SmallVector<ReassociationIndices> groups;
        SmallVector<int64_t> pending;
        bool valid = true;
        size_t si = 0;
        for (size_t ti = 0; ti < topkShape.size() && valid; ++ti) {
          if (si < sliceTy.getShape().size() && topkShape[ti] == sliceTy.getShape()[si]) {
            ReassociationIndices group;
            group.append(pending.begin(), pending.end());
            group.push_back(ti);
            groups.push_back(std::move(group));
            ++si;
          } else if (topkShape[ti] == 1) {
            pending.push_back(ti);
          } else {
            valid = false;
          }
        }
        valid = valid && si == sliceTy.getShape().size();
        if (valid && !pending.empty()) {
          if (groups.empty()) {
            valid = false; // rank-0 slice result
          } else {
            groups.back().append(pending);
          }
        }
        if (!valid || (int64_t)groups.size() != sliceTy.getRank()) {
          return rewriter.notifyMatchFailure(sort, "slice result shape is not "
                                                   "a unit-dim collapse of the "
                                                   "topk shape");
        }
        bool identity = true;
        for (size_t i = 0; i < groups.size(); ++i)
          identity = identity && groups[i].size() == 1 && groups[i][0] == i;
        if (!identity) {
          replacement = rewriter.create<tensor::CollapseShapeOp>(
              loc, replacement, groups);
        }
      }
      // Restore the original element width where the topk result is narrower.
      Type replacementElem = cast<ShapedType>(replacement.getType()).getElementType();
      if (replacementElem != sliceTy.getElementType()) {
        auto widerTy = dyn_cast<IntegerType>(sliceTy.getElementType());
        auto narrowerTy = dyn_cast<IntegerType>(replacementElem);
        if (!widerTy || !narrowerTy || widerTy.getWidth() <= narrowerTy.getWidth()) {
          return rewriter.notifyMatchFailure(sort, "cannot restore the slice "
                                                   "index element type from the "
                                                   "topk result");
        }
        auto widerShapeTy = RankedTensorType::get(sliceTy.getShape(), widerTy);
        Value widerEmpty =
            rewriter.create<tensor::EmptyOp>(loc, sliceTy.getShape(), widerTy);
        int64_t rank = sliceTy.getRank();
        AffineMap idMap = AffineMap::getMultiDimIdentityMap(rank, rewriter.getContext());
        SmallVector<AffineMap> maps{ idMap, idMap };
        SmallVector<utils::IteratorType> iters(rank, utils::IteratorType::parallel);
        replacement = linalg::GenericOp::create(
                          rewriter, loc, widerShapeTy, replacement, widerEmpty, maps,
                          iters, "", "",
                          [widerTy](OpBuilder &b, Location l, ValueRange args) {
                            Value ext =
                                arith::ExtSIOp::create(b, l, widerTy, args[0]);
                            linalg::YieldOp::create(b, l, ext);
                          })
            .getResult(0);
      }
      slices[r].getResult().replaceAllUsesWith(replacement);
      slices[r].erase();
    }
    rewriter.eraseOp(sort);
    return success();
  }
};

struct ScatterOpConversion
    : OpRewritePattern<mlir::torch::TMTensor::ScatterOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::ScatterOp op,
                                PatternRewriter &rewriter) const override {
    auto indicesTy = op.getIndicesType();
    if (!indicesTy.hasRank()) {
      return failure();
    }

    if (indicesTy.isDynamicDim(indicesTy.getRank() - 1)) {
      return rewriter.notifyMatchFailure(op, "number of indices is unknown");
    }

    auto numIndices = indicesTy.getShape().back();
    llvm::SmallVector<int64_t> dimMap(numIndices);
    for (int i = 0; i < numIndices; i++) {
      dimMap[i] = i;
    }

    auto updatesTy = op.getUpdateType();

    // Create a reassociation that drops all unit dims from the indexed portion
    // slice.
    Value updateVal = op.updates();
    SmallVector<int64_t> collapsedShape;
    collapsedShape.push_back(updatesTy.getShape().front());
    if (op.getUpdateSliceRank() > 0) {
      llvm::append_range(collapsedShape,
                         updatesTy.getShape().take_back(
                             op.getUpdateSliceRank() - op.getIndexDepth()));
    }
    if (collapsedShape != updatesTy.getShape()) {
      auto reassocIndices = getReassociationIndicesForCollapse(
          updatesTy.getShape(), collapsedShape);
      if (!reassocIndices.has_value()) {
        return rewriter.notifyMatchFailure(
            op, "failed to compute reassociation indices");
      }
      updateVal = tensor::CollapseShapeOp::create(
          rewriter, op.getLoc(), updateVal, reassocIndices.value());
    }

    Value indicesVal = op.indices();
    auto scatterOp = IREE::LinalgExt::ScatterOp::create(
        rewriter, op.getLoc(), op->getResultTypes(),
        /*updates=*/updateVal, /*indices=*/indicesVal,
        /*mask=*/Value(), /*original=*/op.getOutputs()[0], dimMap,
        op.getUniqueIndices());
    rewriter.inlineRegionBefore(op.getRegion(), scatterOp.getRegion(),
                                scatterOp.getRegion().begin());
    rewriter.replaceOp(op, scatterOp->getResults());
    return success();
  }
};
} // namespace

static SmallVector<AffineMap> getStandardAttentionIndexingMaps(MLIRContext *ctx,
                                                               bool hasMask) {
  AffineExpr m, n, k1, k2;
  bindDims(ctx, m, n, k1, k2);

  auto qMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k1}, ctx);
  auto kMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, k1}, ctx);
  auto vMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {k2, n}, ctx);
  auto sMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, ctx);
  auto rMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, n}, ctx);
  if (hasMask) {
    // Add mask map only if it exists
    auto mMap = AffineMap::get(/*dimCount=*/4, /*symbolCount=*/0, {m, k2}, ctx);
    return {qMap, kMap, vMap, sMap, mMap, rMap};
  }
  return {qMap, kMap, vMap, sMap, rMap};
}

struct AttentionOpConversion
    : OpRewritePattern<mlir::torch::TMTensor::AttentionOp> {
  using Base::Base;
  LogicalResult matchAndRewrite(mlir::torch::TMTensor::AttentionOp op,
                                PatternRewriter &rewriter) const override {
    MLIRContext *ctx = getContext();
    Location loc = op->getLoc();
    Value query = op.getQuery();
    Value key = op.getKey();
    Value value = op.getValue();
    std::optional<Value> optionalMask = op.getAttnMask();

    ShapedType outputType = op.getOutputType();

    SmallVector<Value> dynSizes;
    for (int i = 0, s = outputType.getRank() - 1; i < s; ++i) {
      if (outputType.isDynamicDim(i)) {
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, query, i));
      }
    }

    if (outputType.getShape().back() == ShapedType::kDynamic) {
      dynSizes.push_back(tensor::DimOp::create(rewriter, loc, value,
                                               outputType.getRank() - 1));
    }

    Value result =
        tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                outputType.getElementType(), dynSizes);

    // TODO: This is a hack. This should be replaced with a simple getScale()
    // when support for scaling is plumbed to TMTensor on the torch-mlir side.
    // Until then, we are using the default value used in scaled dot product
    // attention by PyTorch (most models use the default value because it makes
    // the variance of the result of softmax 1 when the mean of Q, K is 0).
    // We use scale = 1 / sqrt(d), where d is the head dimension.
    // See https://paperswithcode.com/method/scaled for more details.
    //
    // TODO: We are currently assuming that head dimension is dim = -1. Once we
    // have support for batch dims using more general indexing maps, we should
    // change this and rely on more general mechanisms.

    Value scale;
    auto scaleAttr = op.getScaleAttr();
    if (scaleAttr) {
      auto targetType = cast<FloatType>(op.getQueryType().getElementType());
      scale = arith::ConstantOp::create(
          rewriter, loc, targetType,
          rewriter.getFloatAttr(targetType, scaleAttr.getValueAsDouble()));
    } else {
      // Compute scale = rsqrt(head_dim) in f32.
      int64_t queryRank = op.getQueryType().getRank();
      Value dimIdx =
          rewriter.createOrFold<tensor::DimOp>(loc, query, queryRank - 1);
      Value dimInt = rewriter.createOrFold<arith::IndexCastOp>(
          loc, rewriter.getI64Type(), dimIdx);
      Value dimFloat = rewriter.createOrFold<arith::SIToFPOp>(
          loc, rewriter.getF32Type(), dimInt);
      scale = rewriter.createOrFold<math::RsqrtOp>(loc, dimFloat);
    }

    // Add batches to standard attention indexing maps.
    SmallVector<AffineMap> indexingMaps =
        getStandardAttentionIndexingMaps(ctx, optionalMask.has_value());

    int64_t numBatches = op.getQueryType().getRank() - 2;
    for (AffineMap &map : indexingMaps) {
      map = map.shiftDims(numBatches);
      if (map.getNumResults() == 0) {
        continue;
      }
      for (int batch : llvm::seq<int>(numBatches)) {
        map = map.insertResult(rewriter.getAffineDimExpr(batch), batch);
      }
    }

    auto attention = IREE::LinalgExt::AttentionOp::create(
        rewriter, loc, result.getType(), query, key, value, scale, result,
        rewriter.getAffineMapArrayAttr(indexingMaps), optionalMask);

    {
      auto *block = rewriter.createBlock(&attention.getRegion());
      OpBuilder::InsertionGuard g(rewriter);
      block->addArgument(rewriter.getF32Type(), loc);
      rewriter.setInsertionPoint(block, block->begin());

      IREE::LinalgExt::YieldOp::create(rewriter, loc, block->getArgument(0));
    }

    rewriter.replaceOp(op, attention.getResult(0));
    return success();
  }
};

namespace {

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

class ConvertTMTensorToLinalgExtPass final
    : public impl::ConvertTMTensorToLinalgExtPassBase<
          ConvertTMTensorToLinalgExtPass> {
public:
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<IREE::LinalgExt::IREELinalgExtDialect>();
    registry.insert<tensor::TensorDialect>();
  }

  void runOnOperation() override {
    MLIRContext *context = &getContext();
    RewritePatternSet patterns(context);
    ConversionTarget target(*context);

#define INSERT_TMTENSOR_CONVERSION_PATTERN(Op)                                 \
  patterns.add<                                                                \
      TMTensorOpConversion<mlir::torch::TMTensor::Op, IREE::LinalgExt::Op>>(   \
      context);

    INSERT_TMTENSOR_CONVERSION_PATTERN(YieldOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(ScanOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(SortOp);
    INSERT_TMTENSOR_CONVERSION_PATTERN(TopkOp);

#undef INSERT_TMTENSOR_CONVERSION_PATTERN

    patterns.add<TopkFromSortPattern>(context);

    patterns.add<ScatterOpConversion>(context);
    patterns.add<AttentionOpConversion>(context);

    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns)))) {
      signalPassFailure();
    }
  }
};
} // namespace

} // namespace mlir::iree_compiler::TorchInput
