#include "LoopUtils.h"

#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/LoopUtils.h"

#define DEBUG_TYPE "LoopUtils"

/**
 * Determine if the static inst is the head of a loop.
 */
// bool isStaticInstLoopHead(const llvm::Loop *Loop,
//                           llvm::Instruction *StaticInst) {
//   assert(Loop != nullptr && "Null loop for isStaticInstLoopHead.");
//   return (Loop->getHeader() == StaticInst->getParent()) &&
//          (StaticInst->getParent()->getFirstNonPHI() == StaticInst);
// }

const std::unordered_set<std::string> LoopUtils::LoopContinuityIgnoredFunctions{
    "exp", "sin", "sqrt", "sqrtf", "acos", "fabs", "abs", "rand", "pow", "log",
    "lgamma", "printf", "omp_get_thread_num", "feof", "m5_work_mark",
    "m5_switch_cpu",
    // SSP Library functions.
    "ssp_config", "ssp_load_i32", "ssp_step", "ssp_end",
    // Specific to spec2017 imagick_s
    // "ReadBlob", "ThrowMagickException", "CloseBlob", "DestroyImage",
};

/**
 * SourceFileName, (Demangled)FunctionName.
 */
const std::list<std::pair<std::string, std::string>>
    LoopUtils::SingleExitFunctions{
        // From GapGraph SSSP.
        {"sssp.cc", "RelaxEdges"},
        {"sssp_check.cc", "RelaxEdges"},
        // From GapGraph BFS_PUSH
        {"bfs_push.cc", ".omp_outlined."},
        {"bfs_push_check.cc", ".omp_outlined."},
    };

std::unordered_map<const llvm::Loop *, bool>
    LoopUtils::MemorizedIsLoopContinuous;

std::string printLoop(const llvm::Loop *Loop) {
  return std::string(Loop->getHeader()->getParent()->getName()) +
         "::" + std::string(Loop->getName());
}

bool LoopUtils::isStaticInstLoopHead(const llvm::Loop *Loop,
                                     const llvm::Instruction *StaticInst) {
  assert(Loop != nullptr && "Null loop for isStaticInstLoopHead.");
  return (Loop->getHeader() == StaticInst->getParent()) &&
         (StaticInst->getParent()->getFirstNonPHI() == StaticInst);
}

bool LoopUtils::isLoopContinuous(const llvm::Loop *Loop) {
  if (MemorizedIsLoopContinuous.count(Loop)) {
    return MemorizedIsLoopContinuous.at(Loop);
  }
  /**
   * Sometimes we want to just enable loops in one function.
   */
  auto Func = Loop->getHeader()->getParent();
  auto FuncUID = Utils::formatLLVMFunc(Func);
  if (FuncUID.find("RelaxEdges") != std::string::npos) {
    LLVM_DEBUG(llvm::dbgs()
               << "Mark LoopContinuous in Func " << FuncUID << '\n');
    MemorizedIsLoopContinuous.emplace(Loop, true);
    return true;
  }

  // Check if there is any calls to unsupported function.
  bool IsContinuous = true;
  for (auto BBIter = Loop->block_begin(), BBEnd = Loop->block_end();
       BBIter != BBEnd; ++BBIter) {
    for (auto InstIter = (*BBIter)->begin(), InstEnd = (*BBIter)->end();
         InstIter != InstEnd; ++InstIter) {
      if (!Utils::isCallOrInvokeInst(&*InstIter)) {
        continue;
      }
      auto *Callee = Utils::getCalledFunction(&*InstIter);
      if (Callee == nullptr) {
        // Indirect call, not continuous.
        LLVM_DEBUG(llvm::dbgs() << "Loop " << printLoop(Loop)
                                << " is statically not continuous because it "
                                   "contains indirect call.\n");
        IsContinuous = false;
        break;
      }
      if (Callee->isIntrinsic()) {
        continue;
      }
      // Check if calling some supported math function.
      if (Callee->isDeclaration()) {
        if (LoopUtils::LoopContinuityIgnoredFunctions.count(
                Callee->getName().str()) != 0) {
          continue;
        }
      }
      LLVM_DEBUG(llvm::dbgs() << "Loop " << printLoop(Loop)
                              << " is statically not continuous because it "
                                 "contains unsupported call to "
                              << Callee->getName() << "\n");
      IsContinuous = false;
      break;
    }
    if (!IsContinuous) {
      break;
    }
  }
  MemorizedIsLoopContinuous.emplace(Loop, IsContinuous);
  return IsContinuous;
}

bool LoopUtils::isLoopRemainderOrEpilogue(const llvm::Loop *Loop) {
  if (llvm::findStringMetadataForLoop(Loop, "llvm.loop.unroll_is_remainder")) {
    return true;
  }
  if (llvm::findStringMetadataForLoop(Loop,
                                      "llvm.loop.vectorize_is_epilogue")) {
    return true;
  }
  return false;
}

std::string LoopUtils::getLoopId(const llvm::Loop *Loop) {
  auto Header = Loop->getHeader();
  auto Func = Header->getParent();
  return Utils::formatLLVMFunc(Func) + "::" + Loop->getName().str();
}

uint64_t LoopUtils::getNumStaticInstInLoop(const llvm::Loop *Loop) {
  uint64_t StaticInsts = 0;
  for (auto BBIter = Loop->block_begin(), BBEnd = Loop->block_end();
       BBIter != BBEnd; ++BBIter) {
    auto BB = *BBIter;
    auto PHIs = BB->phis();
    StaticInsts += BB->size() - std::distance(PHIs.begin(), PHIs.end());
  }
  return StaticInsts;
}

int LoopUtils::countPossiblePath(const llvm::Loop *Loop) {
  std::unordered_set<const llvm::BasicBlock *> OnPathBBs;
  return LoopUtils::countPossiblePathFromBB(Loop, OnPathBBs, Loop->getHeader());
}

int LoopUtils::countPossiblePathFromBB(
    const llvm::Loop *Loop,
    std::unordered_set<const llvm::BasicBlock *> &OnPathBBs,
    const llvm::BasicBlock *CurrentBB) {
  if (!Loop->contains(CurrentBB)) {
    // Exiting the loop.
    return 1;
  }
  int Count = 0;
  OnPathBBs.insert(CurrentBB);
  for (const auto Succ : llvm::successors(CurrentBB)) {
    if (OnPathBBs.count(Succ)) {
      continue;
    }
    Count += LoopUtils::countPossiblePathFromBB(Loop, OnPathBBs, Succ);
  }
  OnPathBBs.erase(CurrentBB);
  return Count;
}

llvm::Instruction *LoopUtils::getUnrollableTerminator(llvm::Loop *Loop) {

  // Basically copies from LoopUnroll.cpp

  auto Header = Loop->getHeader();
  if (Header == nullptr) {
    return nullptr;
  }

  auto LatchBlock = Loop->getLoopLatch();
  if (LatchBlock == nullptr) {
    return nullptr;
  }

  if (!Loop->isSafeToClone()) {
    return nullptr;
  }

  auto BI = llvm::dyn_cast<llvm::BranchInst>(LatchBlock->getTerminator());
  if (!BI || BI->isUnconditional()) {
    return nullptr;
  }

  auto CheckSuccessors = [&](unsigned S1, unsigned S2) {
    return BI->getSuccessor(S1) == Header &&
           !Loop->contains(BI->getSuccessor(S2));
  };

  if (!CheckSuccessors(0, 1) && !CheckSuccessors(1, 0)) {
    return nullptr;
  }

  return BI;
}

const llvm::SCEV *LoopUtils::getTripCountSCEV(llvm::ScalarEvolution *SE,
                                              const llvm::Loop *Loop) {

  auto BackEdgeTakenSCEV = SE->getCouldNotCompute();
  if (SE->hasLoopInvariantBackedgeTakenCount(Loop)) {
    LLVM_DEBUG(llvm::dbgs() << "LoopVariantBackedgeTakenCount for Loop "
                            << LoopUtils::getLoopId(Loop) << "\n");
    BackEdgeTakenSCEV = SE->getBackedgeTakenCount(Loop);
    LLVM_DEBUG({
      llvm::dbgs() << "BackEdgeTakenCount ";
      BackEdgeTakenSCEV->dump();
    });
  }

  const auto &FuncName =
      Utils::getDemangledFunctionName(Loop->getHeader()->getParent());
  if (LoopUtils::isSingleExitLoop(Loop)) {
    // Special case: we simply take the exit count from the the single in-loop
    // predecessor BB of the header.
    llvm::SmallVector<llvm::BasicBlock *, 4> ExitingBBs;
    Loop->getExitingBlocks(ExitingBBs);
    auto HeaderBB = Loop->getHeader();
    llvm::BasicBlock *SingleExitingBB = nullptr;
    for (auto ExitingBB : ExitingBBs) {
      for (auto SuccBB : llvm::successors(ExitingBB)) {
        if (SuccBB == HeaderBB) {
          if (SingleExitingBB) {
            // We already have an single exiting BB.
            llvm::errs() << "Func " << FuncName << " has multi-exit loop "
                         << Loop->getName() << '\n';
            llvm_unreachable("Illegal SingleExit Function.");
          }
          SingleExitingBB = ExitingBB;
          break;
        }
      }
    }
    if (!SingleExitingBB) {
      llvm::errs() << "Func " << FuncName << " has no-exit loop "
                   << Loop->getName() << '\n';
      llvm_unreachable("Illegal SingleExit Function.");
    }
    BackEdgeTakenSCEV = SE->getExitCount(Loop, SingleExitingBB);
    LLVM_DEBUG({
      llvm::dbgs() << "Replace with ExitCount on SingleExitingBB "
                   << SingleExitingBB->getName();
      BackEdgeTakenSCEV->dump();
    });
  }

  if (!llvm::isa<llvm::SCEVCouldNotCompute>(BackEdgeTakenSCEV)) {
    /**
     * We add one to the BackEdgeTakenCount to get trip count.
     */
    auto TripCount = SE->getAddExpr(BackEdgeTakenSCEV,
                                    SE->getOne(BackEdgeTakenSCEV->getType()));
    // hack: Fix the case when TripCount is 0.
    if (auto SMaxSCEV = llvm::dyn_cast<llvm::SCEVSMaxExpr>(TripCount)) {
      bool Fixed = false;
      if (auto OneSCEV =
              llvm::dyn_cast<llvm::SCEVConstant>(SMaxSCEV->getOperand(0))) {
        if (OneSCEV->getAPInt().getZExtValue() == 1) {
          TripCount = SMaxSCEV->getOperand(1);
          Fixed = true;
        }
      }
      if (!Fixed) {
        llvm::errs() << "Wrong TripCount SCEV? ";
        TripCount->print(llvm::errs());
        assert(false);
      }
    }
    return TripCount;
  } else {
    return BackEdgeTakenSCEV;
  }
}

bool LoopUtils::isSingleExitLoop(const llvm::Loop *Loop) {
  std::string FileName;
  if (auto DebugLoc = Loop->getStartLoc()) {
    auto Scope = llvm::cast<llvm::DIScope>(DebugLoc.getScope());
    FileName = Scope->getFilename().str();
  }
  const auto &FuncName =
      Utils::getDemangledFunctionName(Loop->getHeader()->getParent());
  for (const auto &FileFuncPair : LoopUtils::SingleExitFunctions) {
    auto SearchFileName = FileName.find(FileFuncPair.first);
    if (SearchFileName != std::string::npos &&
        FuncName == FileFuncPair.second) {
      // We found a match.
      return true;
    }
  }
  return false;
}

void LoopIterCounter::configure(llvm::Loop *_Loop) {
  this->Loop = _Loop;
  this->Iter = -1;
}

LoopIterCounter::Status LoopIterCounter::count(llvm::Instruction *StaticInst,
                                               int &Iter) {
  assert(this->isConfigured() && "LoopIterCounter is not configured.");
  if (StaticInst == nullptr || (!this->Loop->contains(StaticInst))) {
    // Outside instruction.
    Iter = this->Iter + 1;
    this->Iter = -1;
    return OUT;
  }
  if (LoopUtils::isStaticInstLoopHead(this->Loop, StaticInst)) {
    this->Iter++;
    Iter = this->Iter;
    return NEW_ITER;
  }
  Iter = this->Iter;
  return SAME_ITER;
}

StaticInnerMostLoop::StaticInnerMostLoop(llvm::Loop *_Loop)
    : StaticInstCount(0), Loop(_Loop) {
  assert(Loop->empty() && "Should be inner most loops.");
  this->scheduleBasicBlocksInLoop(this->Loop);
  assert(!this->BBList.empty() && "Empty loops.");

  this->StaticInstCount = LoopUtils::getNumStaticInstInLoop(this->Loop);

  // Create the load/store order map.
  uint32_t LoadStoreIdx = 0;
  for (auto BB : this->BBList) {
    for (auto InstIter = BB->begin(), InstEnd = BB->end(); InstIter != InstEnd;
         ++InstIter) {
      auto Inst = &*InstIter;
      if (llvm::isa<llvm::LoadInst>(Inst) || llvm::isa<llvm::StoreInst>(Inst)) {
        this->LoadStoreOrderMap.emplace(Inst, LoadStoreIdx++);
      }
    }
  }

  // Compute the live in and out.
  this->computeLiveInOutValues(this->Loop);
}

llvm::Instruction *StaticInnerMostLoop::getHeaderNonPhiInst() {
  if (this->BBList.empty()) {
    return nullptr;
  }
  return this->BBList.front()->getFirstNonPHI();
}

void StaticInnerMostLoop::scheduleBasicBlocksInLoop(llvm::Loop *Loop) {
  assert(Loop != nullptr && "Null Loop for scheduleBasicBlocksInLoop.");

  LLVM_DEBUG(llvm::dbgs() << "Schedule basic blocks in loop " << printLoop(Loop)
                          << '\n');

  auto &Schedule = this->BBList;

  std::list<std::pair<llvm::BasicBlock *, bool>> Stack;
  std::unordered_set<llvm::BasicBlock *> Visited;
  std::unordered_set<llvm::BasicBlock *> Scheduled;

  Stack.emplace_back(Loop->getHeader(), false);
  while (!Stack.empty()) {
    auto &Entry = Stack.back();
    auto BB = Entry.first;
    if (Entry.second) {
      // This is the second time we visit BB.
      // We can schedule it.
      if (Scheduled.find(BB) == Scheduled.end()) {
        Schedule.push_front(BB);
        Scheduled.insert(BB);
        LLVM_DEBUG(llvm::dbgs() << "Schedule " << BB->getName() << '\n');
      }
      Stack.pop_back();
      continue;
    }
    // This is the first time we visit BB, do DFS.
    Entry.second = true;
    Visited.insert(BB);
    for (auto Succ : llvm::successors(BB)) {
      if (!Loop->contains(Succ)) {
        continue;
      }
      if (Visited.find(Succ) != Visited.end()) {
        continue;
      }
      LLVM_DEBUG(llvm::dbgs() << "Explore " << Succ->getName() << '\n');
      Stack.emplace_back(Succ, false);
    }
  }

  return;
}

void StaticInnerMostLoop::computeLiveInOutValues(llvm::Loop *Loop) {
  for (auto BB : this->BBList) {
    for (auto InstIter = BB->begin(), InstEnd = BB->end(); InstIter != InstEnd;
         ++InstIter) {
      auto Inst = &*InstIter;
      for (unsigned OperandIdx = 0, NumOperands = Inst->getNumOperands();
           OperandIdx != NumOperands; ++OperandIdx) {
        auto Op = Inst->getOperand(OperandIdx);
        if (auto OpInst = llvm::dyn_cast<llvm::Instruction>(Op)) {
          // This is a register value.
          if (Loop->contains(OpInst)) {
            // This value is generated with in the loop.
            continue;
          }
        }
        // This value is either a global value, function parameter, or outside
        // instruction.
        this->LiveIns.insert(Op);
      }

      // Check the user of this inst.
      for (auto User : Inst->users()) {
        if (auto UserInst = llvm::dyn_cast<llvm::Instruction>(User)) {
          if (!Loop->contains(UserInst)) {
            this->LiveOuts.insert(UserInst);
          }
        } else {
          assert(false && "Other user than instructions?");
        }
      }
    }
  }
}

CachedLoopInfo::~CachedLoopInfo() {
  this->clearAnalysis();
  delete this->DL;
  this->DL = nullptr;
  delete this->TLI;
  this->TLI = nullptr;
}

void CachedLoopInfo::clearAnalysis() {
  this->clearCacheMap(this->SEExpanderCache);
  this->clearCacheMap(this->SECache);
  this->clearCacheMap(this->LICache);
  this->clearCacheMap(this->DTCache);
  this->clearCacheMap(this->ACCache);
  this->UnrollableTerminatorCache.clear();
}

void CachedLoopInfo::clearFuncAnalysis(llvm::Function *Func) {
  this->clearFuncCacheMap(this->SEExpanderCache, Func);
  this->clearFuncCacheMap(this->SECache, Func);
  this->clearFuncCacheMap(this->LICache, Func);
  this->clearFuncCacheMap(this->DTCache, Func);
  this->clearFuncCacheMap(this->ACCache, Func);
  for (auto Iter = this->UnrollableTerminatorCache.begin();
       Iter != this->UnrollableTerminatorCache.end();) {
    auto Loop = Iter->first;
    if (Loop->getHeader()->getParent() == Func) {
      Iter = this->UnrollableTerminatorCache.erase(Iter);
    } else {
      ++Iter;
    }
  }
}

llvm::AssumptionCache *
CachedLoopInfo::getAssumptionCache(llvm::Function *Func) {
  auto Iter = this->ACCache.find(Func);
  if (Iter == this->ACCache.end()) {
    Iter = this->ACCache.emplace(Func, new llvm::AssumptionCache(*Func)).first;
  }
  return Iter->second;
}

llvm::LoopInfo *CachedLoopInfo::getLoopInfo(llvm::Function *Func) {
  auto Iter = this->LICache.find(Func);
  if (Iter == this->LICache.end()) {
    Iter = this->LICache
               .emplace(Func, new llvm::LoopInfo(*this->getDominatorTree(Func)))
               .first;
  }
  return Iter->second;
}

llvm::DominatorTree *CachedLoopInfo::getDominatorTree(llvm::Function *Func) {
  auto Iter = this->DTCache.find(Func);
  if (Iter == this->DTCache.end()) {
    auto DT = new llvm::DominatorTree();
    DT->recalculate(*Func);
    Iter = this->DTCache.emplace(Func, DT).first;
  }
  return Iter->second;
}

llvm::ScalarEvolution *
CachedLoopInfo::getScalarEvolution(llvm::Function *Func) {
  auto Iter = this->SECache.find(Func);
  if (Iter == this->SECache.end()) {
    auto SE = new llvm::ScalarEvolution(
        *Func, *this->TLI, *this->getAssumptionCache(Func),
        *this->getDominatorTree(Func), *this->getLoopInfo(Func));
    Iter = this->SECache.emplace(Func, SE).first;
  }
  return Iter->second;
}

llvm::SCEVExpander *CachedLoopInfo::getSCEVExpander(llvm::Function *Func) {
  auto Iter = this->SEExpanderCache.find(Func);
  if (Iter == this->SEExpanderCache.end()) {
    auto SEExpander =
        new llvm::SCEVExpander(*this->getScalarEvolution(Func), *this->DL, "");
    Iter = this->SEExpanderCache.emplace(Func, SEExpander).first;
  }
  return Iter->second;
}

llvm::TargetTransformInfo
CachedLoopInfo::getTargetTransformInfo(llvm::Function *Func) {
  llvm::TargetIRAnalysis TIRA;
  llvm::FunctionAnalysisManager DummyFAM;
  return TIRA.run(*Func, DummyFAM);
}

llvm::Instruction *CachedLoopInfo::getUnrollableTerminator(llvm::Loop *Loop) {
  auto Iter = this->UnrollableTerminatorCache.find(Loop);
  if (Iter == this->UnrollableTerminatorCache.end()) {
    Iter = this->UnrollableTerminatorCache
               .emplace(Loop, LoopUtils::getUnrollableTerminator(Loop))
               .first;
  }
  return Iter->second;
}

#undef DEBUG_TYPE