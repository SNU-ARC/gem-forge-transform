#include "DynamicLoopTree.h"
#include "Replay.h"
#include "Utils.h"
#include "stream/InlineContext.h"
#include "stream/InlineContextStream.h"
#include "trace/ProfileParser.h"

#include "llvm/Analysis/ScalarEvolutionExpressions.h"

#define DEBUG_TYPE "InlineContextStreamPass"
#if !defined(LLVM_DEBUG) && defined(DEBUG)
#define LLVM_DEBUG DEBUG
#endif

namespace {

class ScalarUIntVar {
public:
  ScalarUIntVar(const std::string &_name) : Val(0), name(_name) {}
  uint64_t Val;
  std::string name;
  void print(llvm::raw_ostream &O) const {
    O << this->name << ": " << this->Val << '\n';
  }
  void print(std::ostream &O) const {
    O << this->name << ": " << this->Val << '\n';
  }
};

class StreamLoadInst : public DynamicInstruction {
public:
  StreamLoadInst(DynamicId Id) : DynamicInstruction() {
    // Hack: override my new id with the id of the original load inst.
    this->Id = Id;
  }
  std::string getOpName() const override { return "stream-load"; }
};

class StreamStoreInst : public DynamicInstruction {
public:
  StreamStoreInst(DynamicId Id) : DynamicInstruction() {
    // Same hack.
    this->Id = Id;
  }
  std::string getOpName() const override { return "stream-store"; }
};

class InlineContextStreamPass : public ReplayTrace {
public:
  static char ID;

#define INIT_VAR(var) var(#var)
  InlineContextStreamPass()
      : ReplayTrace(ID), INIT_VAR(DynInstCount), INIT_VAR(DynMemInstCount),
        INIT_VAR(TracedDynInstCount), INIT_VAR(AddRecLoadCount),
        INIT_VAR(AddRecStoreCount), INIT_VAR(ConstantCount),
        INIT_VAR(LinearCount), INIT_VAR(QuardricCount), INIT_VAR(IndirectCount),
        INIT_VAR(StreamCount), INIT_VAR(RemovedAddrInstCount) {}
#undef INIT_VAR

protected:
  bool initialize(llvm::Module &Module) override {
    bool Ret = ReplayTrace::initialize(Module);

    // this->State = SEARCHING;
    // this->LoopTree = nullptr;

    this->MemorizedMemoryAccess.clear();

    return Ret;
  }

  bool finalize(llvm::Module &Module) override {

    // this->State = SEARCHING;
    // if (this->LoopTree != nullptr) {
    //   delete this->LoopTree;
    //   this->LoopTree = nullptr;
    // }

    return ReplayTrace::finalize(Module);
  }

  void dumpStats(std::ostream &O) override;

  void transform() override;

  /**
   * A map from ContextLoop to a list of active streams that is being analyzed
   * at that level of loop.
   * The list stores only reference, as the object is stored in InstStreamMap.
   */
  using ActiveStreamMap = std::unordered_map<
      ContextLoop, std::unordered_map<ContextInst, InlineContextStream *>>;

  /**
   * Analyze the stream pattern in the first pass.
   */
  void analyzeStream();

  void addAccess(const InlineContextPtr &CurrentContext,
                 const std::list<ContextLoop> &LoopStack,
                 ActiveStreamMap &ActiveStreams,
                 DynamicInstruction *DynamicInst);
  void endIter(const std::list<ContextLoop> &LoopStack,
               ActiveStreamMap &ActiveStreams);
  void pushLoopStack(const InlineContextPtr &CurrentContext,
                     std::list<ContextLoop> &LoopStack,
                     ActiveStreamMap &ActiveStreams,
                     const ContextLoop &NewContextLoop);
  void popLoopStack(std::list<ContextLoop> &LoopStack,
                    ActiveStreamMap &ActiveStreams);
  void activateStream(ActiveStreamMap &ActiveStreams, const ContextInst &CInst);

  bool initializeStreamIfNecessary(const std::list<ContextLoop> &LoopStack,
                                   const ContextInst &CInst);

  // /**
  //  * Replace the memory access with our stream interface in the second pass.
  //  */
  // void transformStream();

  // bool isLoopStream(llvm::Loop *Loop);

  // /**
  //  * Initialize the MemInstToStreamMap and MemorizedMemoryAccess.
  //  */
  // void initializeLoopStream(llvm::Loop *Loop);

  /**
   * For all the analyzed stream, choose the loop level for optimization.
   */
  void chooseStream();

  // void computeMemoryAccessPattern(DynamicLoopIteration *LoopTree);

  std::string classifyStream(const InlineContextStream &S) const;

  ProfileParser Profile;

  /**
   * A map from ContextInst to a list of streams.
   * Each stream contains the analysis result of the ContextInst in different
   * loop levels. Sorted from inner most loop to outer loops.
   */
  std::unordered_map<ContextInst, std::list<InlineContextStream>> InstStreamMap;

  static void DEBUG_LOOP_STACK(const std::list<ContextLoop> &LoopStack);
  static void DEBUG_ACTIVE_STREAM_MAP(const std::list<ContextLoop> &LoopStack,
                                      const ActiveStreamMap &ActiveStreams);

  enum LoopStatusT {
    CONTINUOUS,
    INLINE_CONTINUOUS,
    INDIRECT_CONTINUOUS,
    INCONTINUOUS,
    RECURSIVE,
  };

  static std::string formatLoopStatusT(const LoopStatusT Status) {
    switch (Status) {
    case CONTINUOUS:
      return "CONTINUOUS";
    case INLINE_CONTINUOUS:
      return "INLINE_CONTINUOUS";
    case INDIRECT_CONTINUOUS:
      return "INDIRECT_CONTINUOUS";
    case INCONTINUOUS:
      return "INCONTINUOUS";
    case RECURSIVE:
      return "RECURSIVE";
    default: {
      llvm_unreachable("Illegal LoopStatusT for formating.");
      break;
    }
    }
  }

  static LoopStatusT initLoopStatusT(const LoopStatusT NestedLoopStatus) {
    // if (NestedLoopStatus == RECURSIVE) {
    //   return RECURSIVE;
    // } else {
    //   return CONTINUOUS;
    // }
    return CONTINUOUS;
  }

  static LoopStatusT updateLoopStatusT(const LoopStatusT CurrentStatus,
                                       const LoopStatusT NewStatus) {
    return CurrentStatus > NewStatus ? CurrentStatus : NewStatus;
  }

  /**
   * A map from ContextLoop to its status.
   * Currently the status can be CONTINUOUS, INCONTINUOUS, RECURSIVE.
   * TODO: Rewrite this part.
   */
  std::unordered_map<ContextLoop, LoopStatusT> LoopStatus;

  // /**
  //  * A map from memory access instruction to a list of streams.
  //  * Each stream contains the analyzation result of the instruction in
  //  different
  //  * loop levels.
  //  */
  // std::unordered_map<llvm::Instruction *, std::list<Stream>>
  // MemInstToStreamMap;

  // /**
  //  * A map from memory access instruction to the chosen stream at a
  //  specific
  //  * loop level. The stream is actually stored in MemInstToStreamMap. Be
  //  careful
  //  * with the life span.
  //  */
  // std::unordered_map<llvm::Instruction *, Stream *>
  // MemInstToChosenStreamMap;

  // /**
  //  * A set contains the removed address compute intsruction.
  //  */
  // std::unordered_set<llvm::Instruction *>
  // RemovableAddrComputeInstSet;

  // DynamicLoopIteration *LoopTree;

  // std::unordered_map<llvm::Loop *, bool> MemorizedLoopStream;
  std::unordered_map<const llvm::Loop *,
                     std::unordered_set<llvm::Instruction *>>
      MemorizedMemoryAccess;

  // std::unordered_map<llvm::Loop *, size_t> LoopOngoingIters;

  /*******************************************************
   * Statistics.
   *******************************************************/
  ScalarUIntVar DynInstCount;
  ScalarUIntVar DynMemInstCount;
  ScalarUIntVar TracedDynInstCount;

  ScalarUIntVar AddRecLoadCount;
  ScalarUIntVar AddRecStoreCount;
  ScalarUIntVar ConstantCount;
  ScalarUIntVar LinearCount;
  ScalarUIntVar QuardricCount;
  ScalarUIntVar IndirectCount;
  ScalarUIntVar StreamCount;
  ScalarUIntVar RemovedAddrInstCount;

  // A map to store the all the memory access instruction.
  std::unordered_map<ContextInst, uint64_t> MemAccessInstCount;
}; // namespace

std::string
InlineContextStreamPass::classifyStream(const InlineContextStream &S) const {

  // First check the loop status.
  const auto &CLoop = S.getContextLoop();
  auto Status = this->LoopStatus.at(CLoop);
  if (Status == RECURSIVE) {
    return "RECURSIVE";
  } else if (Status == INCONTINUOUS) {
    return "INCONTINUOUS";
  } else if (Status == INDIRECT_CONTINUOUS) {
    return "INDIRECT_CONTINUOUS";
  }

  // The loop is continuous or inline continous.
  // We can further classify the stream now.
  if (S.getNumBaseLoads() > 1) {
    return "MULTI_BASE";
  } else if (S.getNumBaseLoads() == 1) {
    const auto &BaseLoad = *S.getBaseLoads().begin();
    if (BaseLoad == S.getContextInst()) {
      // Chasing myself.
      return "POINTER_CHASE";
    }
    auto BaseStreamsIter = this->InstStreamMap.find(BaseLoad);
    if (BaseStreamsIter != this->InstStreamMap.end()) {
      const auto &BaseStreams = BaseStreamsIter->second;
      for (const auto &BaseStream : BaseStreams) {
        if (BaseStream.getContextLoop() == S.getContextLoop()) {
          if (BaseStream.getNumBaseLoads() != 0) {
            return "CHAIN_BASE";
          }
          if (BaseStream.getPattern().computed()) {
            auto Pattern = BaseStream.getPattern().getPattern().ValPattern;
            if (Pattern <= StreamPattern::ValuePattern::QUARDRIC) {
              return "AFFINE_BASE";
            }
          }
        }
      }
    }
    return "RANDOM_BASE";
  } else {
    if (S.getPattern().computed()) {
      auto Pattern = S.getPattern().getPattern().ValPattern;
      if (Pattern <= StreamPattern::ValuePattern::QUARDRIC) {
        return "AFFINE";
      } else {
        return "RANDOM";
      }
    } else {
      return "RANDOM";
    }
  }
}

void InlineContextStreamPass::dumpStats(std::ostream &O) {
  O << "--------------- Stats -----------------\n";
  this->DynInstCount.print(O);
  this->DynMemInstCount.print(O);
  this->TracedDynInstCount.print(O);
  this->AddRecLoadCount.print(O);
  this->AddRecStoreCount.print(O);
  this->ConstantCount.print(O);
  this->LinearCount.print(O);
  this->QuardricCount.print(O);
  this->IndirectCount.print(O);
  this->StreamCount.print(O);
  this->RemovedAddrInstCount.print(O);

  /**
   * Loop
   * Inst
   * AddrPat
   * AccPat
   * Iters
   * Accs
   * Updates
   * StreamCount
   * LoopLevel
   * BaseLoads
   * StreamClass
   * Footprint
   * AddrInsts
   * AliasInsts
   * Qualfied
   * Chosen
   */
  O << "--------------- Stream -----------------\n";
  for (auto &MemInstStreamEntry : this->InstStreamMap) {
    for (auto &Stream : MemInstStreamEntry.second) {
      O << Stream.getContextLoop().format();
      O << ' ' << Stream.getContextInst().format();

      // std::string StreamClass = this->classifyStream(Stream);

      std::string AddrPattern = "NOT_COMPUTED";
      std::string AccPattern = "NOT_COMPUTED";
      size_t Iters = Stream.getTotalIters();
      size_t Accesses = Stream.getTotalAccesses();
      size_t Updates = 0;
      size_t StreamCount = Stream.getTotalStreams();
      size_t LoopLevel = 0;
      size_t BaseLoads = Stream.getNumBaseLoads();

      std::string StreamClass = this->classifyStream(Stream);
      size_t Footprint = 0;
      size_t AddrInsts = Stream.getNumAddrInsts();

      std::string Qualified = "NO";

      std::string Chosen = "NO";
      // auto Iter =
      // this->MemInstToChosenStreamMap.find(MemInstStreamEntry.first); if (Iter
      // != this->MemInstToChosenStreamMap.end()) {
      //   if (Iter->second == &Stream) {
      //     Chosen = "YES";
      //   }
      // }

      if (Stream.getPattern().computed()) {
        const auto &Pattern = Stream.getPattern().getPattern();
        AddrPattern = StreamPattern::formatValuePattern(Pattern.ValPattern);
        AccPattern = StreamPattern::formatAccessPattern(Pattern.AccPattern);
        Updates = Pattern.Updates;
        Footprint = Stream.getFootprint().getNumCacheLinesAccessed();
      }
      O << ' ' << AddrPattern;
      O << ' ' << AccPattern;
      O << ' ' << Iters;
      O << ' ' << Accesses;
      O << ' ' << Updates;
      O << ' ' << StreamCount;
      O << ' ' << LoopLevel;
      O << ' ' << BaseLoads;
      O << ' ' << StreamClass;
      O << ' ' << Footprint;
      O << ' ' << AddrInsts;
      O << ' ' << -1; // AliasInsts.
      O << ' ' << Qualified;
      O << ' ' << Chosen;
      O << '\n';
    }
  }

  // Also dump statistics for all the unanalyzed memory accesses.
  size_t TotalAccesses = 0;
  for (const auto &Entry : this->MemAccessInstCount) {
    const auto &CInst = Entry.first;
    TotalAccesses += Entry.second;
    if (this->InstStreamMap.count(CInst) != 0) {
      // This instruction is already dumped.
      if (!this->InstStreamMap.at(CInst).empty()) {
        continue;
      }
    }
    std::string LoopId = "UNKNOWN";
    auto LI = this->CachedLI->getLoopInfo(
        const_cast<llvm::Function *>(CInst.Inst->getFunction()));
    auto Loop = LI->getLoopFor(CInst.Inst->getParent());
    if (Loop != nullptr) {
      LoopId = LoopUtils::getLoopId(Loop);
    }
    O << LoopId;
    O << ' ' << CInst.format(); // Inst
    O << ' ' << "NOT_STREAM";   // Address pattern
    O << ' ' << "NOT_STREAM";   // Access pattern
    O << ' ' << 0;              // Iters
    O << ' ' << Entry.second;   // Accesses
    O << ' ' << 0;              // Updates
    O << ' ' << 0;              // StreamCount
    O << ' ' << -1;             // LoopLevel
    O << ' ' << -1;             // BaseLoads
    O << ' ' << "NOT_STREAM";   // Stream class
    O << ' ' << 0;              // Footprint
    O << ' ' << -1;             // AddrInsts
    O << ' ' << -1;             // AliasInsts
    O << ' ' << "NO";           // Qualified
    O << ' ' << "NO" << '\n';   // Chosen
  }

  assert(TotalAccesses == this->DynMemInstCount.Val &&
         "Mismatch total accesses.");

  O << "--------------- Loop -----------------\n";
  for (const auto &LoopStatusEntry : this->LoopStatus) {
    O << LoopStatusEntry.first.format() << ' '
      << InlineContextStreamPass::formatLoopStatusT(LoopStatusEntry.second)
      << '\n';
  }
}

void InlineContextStreamPass::transform() {
  // First pass.
  this->analyzeStream();

  for (auto &MemInstStreamEntry : this->InstStreamMap) {
    for (auto &Stream : MemInstStreamEntry.second) {
      // if (!Stream.getPattern().computed()) {
      //   llvm::errs() << LoopUtils::getLoopId(Stream.getLoop()) << " "
      //                << Utils::formatLLVMInst(Stream.getMemInst())
      //                << " is not computed.\n";
      // } else {
      //  LLVM_DEBUG(llvm::dbgs() << LoopUtils::getLoopId(Stream.getLoop()) << "
      //  "
      //                      << Utils::formatLLVMInst(Stream.getMemInst())
      //                      << " is computed.\n");
      // }
      Stream.finalizePattern();
    }
  }

  // delete this->Trace;
  // this->Trace = new DataGraph(this->Module, this->CachedLI, this->CachedPDF,
  // this->CachedLU, this->DGDetailLevel);

  // Second pass.
  // this->transformStream();
}

// void InlineContextStreamPass::transformStream() {

//   // Grows at the end.
//   std::list<llvm::Loop *> RegionStack;

//  LLVM_DEBUG(llvm::dbgs() << "Stream: Start Pass 2.\n");

//   while (true) {

//     auto NewInstIter = this->Trace->loadOneDynamicInst();

//     if (NewInstIter == this->Trace->DynamicInstructionList.end()) {
//       // We at the end of the loop. Commit everything.
//       while (!this->Trace->DynamicInstructionList.empty()) {
//         // Serialize in the second pass.
//         this->Serializer->serialize(this->Trace->DynamicInstructionList.front(),
//                                     this->Trace);
//         this->Trace->commitOneDynamicInst();
//       }
//       break;
//     }

//     // This is a new instruction.
//     auto NewDynamicInst = *NewInstIter;
//     auto NewStaticInst = NewDynamicInst->getStaticInstruction();
//     assert(NewStaticInst != nullptr && "Invalid static llvm instruction.");

//     auto LI = this->CachedLI->getLoopInfo(NewStaticInst->getFunction());
//     auto NewLoop = LI->getLoopFor(NewStaticInst->getParent());

//     bool IsAtHeadOfCandidate = false;

//     if (NewLoop != nullptr) {
//       if (this->isLoopStream(NewLoop)) {
//         IsAtHeadOfCandidate =
//             LoopUtils::isStaticInstLoopHead(NewLoop, NewStaticInst);
//       } else {
//         NewLoop = nullptr;
//       }
//     }

//     // Pop all the exited region.
//     while (!RegionStack.empty() &&
//            !RegionStack.back()->contains(NewStaticInst)) {
//       RegionStack.pop_back();
//     }

//     // Enter new region.
//     if (NewLoop != nullptr && IsAtHeadOfCandidate) {
//       RegionStack.push_back(NewLoop);
//     }

//     // Do the transformation.
//     // Replace the stream load/store instruction.
//     // Remove the removable address compute instruction.
//     if (this->RemovableAddrComputeInstSet.count(NewStaticInst) != 0) {

//       // In order to maintain some sort of dependence chain, I will update
//       the
//       // register dependence look up table here.
//       // This works as this is the last instruction in the stream so far, so
//       the
//       // translateRegisterDependence will automatically pick up the
//       dependence. std::list<DynamicInstruction::DynamicId> RegDeps; for
//       (const auto &RegDep :
//            this->Trace->RegDeps.at(NewDynamicInst->getId())) {
//         RegDeps.push_back(RegDep.second);
//       }
//       this->Trace->DynamicFrameStack.front().updateRegisterDependenceLookUpMap(
//           NewStaticInst, std::move(RegDeps));

//       // Remove from the dependence map and so on.
//       this->Trace->commitDynamicInst(NewDynamicInst->getId());

//       // Remove from the list.
//       this->Trace->DynamicInstructionList.erase(NewInstIter);

//       this->RemovedAddrInstCount.Val++;

//     } else if (this->MemInstToChosenStreamMap.count(NewStaticInst) != 0) {
// // #define REPLACE_FOR_IDEAL_CACHE
// #ifdef REPLACE_FOR_IDEAL_CACHE
//       bool IsLoad = llvm::isa<llvm::LoadInst>(NewStaticInst);
//       DynamicInstruction *NewStreamInst = nullptr;
//       if (IsLoad) {
//         NewStreamInst = new StreamLoadInst(NewDynamicInst->getId());
//       } else {
//         NewStreamInst = new StreamStoreInst(NewDynamicInst->getId());
//       }

//       // Replace the original memory access with stream access.
//       *NewInstIter = NewStreamInst;

//       // Remember to delete the original one.
//       delete NewDynamicInst;
// #endif
//     }

//     // Serialize the instruction.
//     // Set a sufficient large buffer size so that the conditional branch
//     // instruction can find the next basic block.
//     if (this->Trace->DynamicInstructionList.size() == 100) {
//       this->Serializer->serialize(this->Trace->DynamicInstructionList.front(),
//                                   this->Trace);
//       this->Trace->commitOneDynamicInst();
//     }
//   }

//  LLVM_DEBUG(llvm::dbgs() << "Stream: End.\n");
// }

// void InlineContextStreamPass::computeMemoryAccessPattern(DynamicLoopIteration
// *LoopTree) {
//   assert(LoopTree != nullptr &&
//          "Null LoopTree when trying to compute memory access pattern.");

//   auto Loop = LoopTree->getLoop();
//   auto TripCount = LoopTree->countIter();

//   const auto &MemAccess = this->MemorizedMemoryAccess.at(Loop);

//   auto Iter = LoopTree;
//   for (size_t Trip = 0; Trip < TripCount; ++Trip) {

//     // Count for all the nested loop recursively.
//     for (auto NestIter = Loop->begin(), NestEnd = Loop->end();
//          NestIter != NestEnd; ++NestIter) {
//       auto ChildLoop = *NestIter;
//       this->computeMemoryAccessPattern(Iter->getChildIter(ChildLoop));
//     }

//     // Compute pattern for memory access within this loop.
//     for (auto MemAccessInst : MemAccess) {
//       auto DynamicInst = Iter->getDynamicInst(MemAccessInst);
//       auto &StreamList = this->MemInstToStreamMap.at(MemAccessInst);
//       if (DynamicInst == nullptr) {
//         // This memory access is not performed in this iteration.
//         for (auto &Stream : StreamList) {
//           Stream.addMissingAccess();
//         }
//       } else {
//         uint64_t Addr = 0;
//         if (llvm::isa<llvm::LoadInst>(MemAccessInst)) {
//           Addr = DynamicInst->DynamicOperands[0]->getAddr();
//         } else {
//           Addr = DynamicInst->DynamicOperands[1]->getAddr();
//         }
//         for (auto &Stream : StreamList) {
//           Stream.addAccess(Addr);
//         }
//       }
//     }

//     Iter = Iter->getNextIter();
//   }

//   if (Iter == nullptr) {
//     // This is the end of the loop, we should end the stream.
//     size_t PreviousIters = 0;
//     if (this->LoopOngoingIters.find(Loop) != this->LoopOngoingIters.end()) {
//       PreviousIters = this->LoopOngoingIters.at(Loop);
//     }
//     this->LoopOngoingIters.erase(Loop);
//     // We have to end the stream for all memory instruction with in the loop.
//     // TODO: Memorize this?
//     for (auto BBIter = Loop->block_begin(), BBEnd = Loop->block_end();
//          BBIter != BBEnd; ++BBIter) {
//       auto BB = *BBIter;
//       for (auto InstIter = BB->begin(), InstEnd = BB->end();
//            InstIter != InstEnd; ++InstIter) {
//         auto Inst = &*InstIter;
//         if ((!llvm::isa<llvm::LoadInst>(Inst)) &&
//             (!llvm::isa<llvm::StoreInst>(Inst))) {
//           continue;
//         }
//         auto &StreamList = this->MemInstToStreamMap.at(Inst);
//         for (auto &Stream : StreamList) {
//           if (Stream.getLoop() == Loop) {
//             //LLVM_DEBUG(llvm::dbgs()
//             //       << "Ending stream at loop " <<
//             LoopUtils::getLoopId(Loop)
//             //       << " inst " << Utils::formatLLVMInst(Inst) << '\n');
//             //LLVM_DEBUG(llvm::dbgs()
//             //       << "Inner most loop is "
//             //       << LoopUtils::getLoopId(
//             //              this->CachedLI->getLoopInfo(Inst->getFunction())
//             //                  ->getLoopFor(Inst->getParent()))
//             //       << '\n');
//             //LLVM_DEBUG(llvm::dbgs()
//             //       << "Inst is memorized? " << MemAccess.count(Inst) <<
//             '\n'); Stream.endStream(); break;
//           }
//         }
//       }
//     }
//   } else {
//     // This loop is still going on.
//     size_t PreviousIters = 0;
//     if (this->LoopOngoingIters.find(Loop) != this->LoopOngoingIters.end()) {
//       PreviousIters = this->LoopOngoingIters.at(Loop);
//     }
//     this->LoopOngoingIters[Loop] = PreviousIters + TripCount;
//   }
// }

#define DEBUG_TARGET_CINST                                                     \
  "Main->spec_qsort::bb711::1(call)->spec_qsort::bb197::tmp201(call)->cost_"   \
  "compare::bb::tmp(load)"

#define DEBUG_TARGET_CLOOP                                                     \
  "Main->lib_link_all::bb399::0(call)->lib_link_all::bb119"

void InlineContextStreamPass::DEBUG_LOOP_STACK(
    const std::list<ContextLoop> &LoopStack) {
  LLVM_DEBUG(llvm::dbgs() << "LoopStack Top --------------------------\n");
  for (auto Iter = LoopStack.rbegin(), End = LoopStack.rend(); Iter != End;
       ++Iter) {
    LLVM_DEBUG(llvm::dbgs() << Iter->beautify() << "\n----------------\n");
  }
  LLVM_DEBUG(llvm::dbgs() << "LoopStack Bottom -----------------------\n");
}

void InlineContextStreamPass::DEBUG_ACTIVE_STREAM_MAP(
    const std::list<ContextLoop> &LoopStack,
    const ActiveStreamMap &ActiveStreams) {
  LLVM_DEBUG(llvm::dbgs() << "ActiveStreams Top ----------------------\n");
  for (auto Iter = LoopStack.rbegin(), End = LoopStack.rend(); Iter != End;
       ++Iter) {
    auto ActiveStreamsAtCLoopIter = ActiveStreams.find(*Iter);
    if (ActiveStreamsAtCLoopIter != ActiveStreams.end()) {
      LLVM_DEBUG(llvm::dbgs()
                 << ActiveStreamsAtCLoopIter->second.size() << '\n');
    } else {
      LLVM_DEBUG(llvm::dbgs() << 0 << '\n');
    }
  }
  LLVM_DEBUG(llvm::dbgs() << "ActiveStreams Bottom -------------------\n");
}

bool InlineContextStreamPass::initializeStreamIfNecessary(
    const std::list<ContextLoop> &LoopStack, const ContextInst &CInst) {

  // Initialize the global count.
  {
    auto Iter = this->MemAccessInstCount.find(CInst);
    if (Iter == this->MemAccessInstCount.end()) {
      this->MemAccessInstCount.emplace(CInst, 0);
    }
  }

  // Check if this is the first time we encounter this access.
  auto Iter = this->InstStreamMap.find(CInst);
  if (Iter == this->InstStreamMap.end()) {
    Iter = this->InstStreamMap.emplace(CInst, std::list<InlineContextStream>())
               .first;
  }

#ifdef DEBUG_TARGET_CINST
  if (CInst.format() == DEBUG_TARGET_CINST) {
    LLVM_DEBUG(llvm::dbgs() << "Handling initializing stream for "
                            << CInst.format() << "\n");
    LLVM_DEBUG(InlineContextStreamPass::DEBUG_LOOP_STACK(LoopStack));
  }
#endif

  auto &Streams = Iter->second;
  auto CLoopIter = LoopStack.rbegin();
  for (const auto &Stream : Streams) {
    if (CLoopIter == LoopStack.rend()) {
      break;
    }
    if (Stream.getContextLoop() != (*CLoopIter)) {
      LLVM_DEBUG(llvm::dbgs() << "CInst --------------\n");
      LLVM_DEBUG(llvm::dbgs() << CInst.beautify() << '\n');
      LLVM_DEBUG(InlineContextStreamPass::DEBUG_LOOP_STACK(LoopStack));
      LLVM_DEBUG(llvm::dbgs() << "Stream Context --------------\n");
      LLVM_DEBUG(llvm::dbgs() << Stream.getContextLoop().beautify() << '\n');
      LLVM_DEBUG(llvm::dbgs() << "Current Loop --------------\n");
      LLVM_DEBUG(llvm::dbgs() << CLoopIter->beautify() << '\n');
    }
    assert(Stream.getContextLoop() == (*CLoopIter) &&
           "Mismatch initialized stream.");
    ++CLoopIter;
  }

  // Initialize the remaining loops.
  while (CLoopIter != LoopStack.rend()) {
#ifdef DEBUG_TARGET_CINST
    if (CInst.format() == DEBUG_TARGET_CINST) {
      LLVM_DEBUG(llvm::dbgs()
                 << "Initialize stream for " DEBUG_TARGET_CINST " at level "
                 << CLoopIter->beautify() << '\n');
    }
#endif

    auto LoopLevel = Streams.size();
    Streams.emplace_back(CInst, *CLoopIter, LoopLevel);
    ++CLoopIter;
  }

  return true;
}

void InlineContextStreamPass::activateStream(ActiveStreamMap &ActiveStreams,
                                             const ContextInst &CInst) {
  auto InstStreamIter = this->InstStreamMap.find(CInst);
  assert(InstStreamIter != this->InstStreamMap.end() &&
         "Failed to look up streams to activate.");
  /**
   * This two fold loop is very expensive.
   * Can we optimize it?
   */
  for (auto &S : InstStreamIter->second) {
    const auto &CLoop = S.getContextLoop();
    auto ActiveStreamsIter = ActiveStreams.find(CLoop);
    if (ActiveStreamsIter == ActiveStreams.end()) {
      ActiveStreamsIter =
          ActiveStreams
              .emplace(CLoop,
                       std::unordered_map<ContextInst, InlineContextStream *>())
              .first;
    }
    auto &ActiveStreamsInstMap = ActiveStreamsIter->second;

    auto ActiveStreamsInstMapIter = ActiveStreamsInstMap.find(CInst);
    if (ActiveStreamsInstMap.count(CInst) == 0) {
      ActiveStreamsInstMap.emplace(CInst, &S);
    } else {
      // Otherwise, the stream is already activated.
      // Ignore this case.
    }
  }
}

void InlineContextStreamPass::pushLoopStack(
    const InlineContextPtr &CurrentContext, std::list<ContextLoop> &LoopStack,
    ActiveStreamMap &ActiveStreams, const ContextLoop &NewContextLoop) {

  // Compute the loop status for the new loop.
  auto NewLoopStatus = CONTINUOUS;
  if (!LoopStack.empty()) {
    const auto &PrevContextLoop = LoopStack.back();
    const auto &PrevLoopStatus = this->LoopStatus.at(PrevContextLoop);
    NewLoopStatus = InlineContextStreamPass::initLoopStatusT(PrevLoopStatus);
  }
  // Manage the loop status.
  if (this->LoopStatus.count(NewContextLoop) == 0) {
    this->LoopStatus.emplace(NewContextLoop, NewLoopStatus);
  } else {
    this->LoopStatus.at(NewContextLoop) =
        InlineContextStreamPass::updateLoopStatusT(
            this->LoopStatus.at(NewContextLoop), NewLoopStatus);
  }
  // LLVM_DEBUG(llvm::dbgs() << "Push loop stack with status "
  //                    << InlineContextStreamPass::formatLoopStatusT(
  //                           this->LoopStatus.at(NewContextLoop))
  //                    << '\n');

#ifdef DEBUG_TARGET_CLOOP
  if (NewContextLoop.format() == DEBUG_TARGET_CLOOP) {
    LLVM_DEBUG(llvm::dbgs() << "Push loop stack with context loop:\n"
                            << NewContextLoop.beautify() << '\n');
  }
#endif

  LoopStack.emplace_back(NewContextLoop);
  // Iterate through all the memory access instructions at this level of loop.
  // Memorize this.
  auto Iter = this->MemorizedMemoryAccess.find(NewContextLoop.Loop);
  if (Iter == this->MemorizedMemoryAccess.end()) {

    // if (LoopUtils::getLoopId(NewContextLoop.Loop) == "VerticalFilter::bb268")
    // {
    //  LLVM_DEBUG(llvm::dbgs()
    //         << "Initialize memory accesses for loop
    //         VerticalFilter::bb268\n");
    // }

    Iter = this->MemorizedMemoryAccess
               .emplace(NewContextLoop.Loop,
                        std::unordered_set<llvm::Instruction *>())
               .first;

    const auto &SubLoops = NewContextLoop.Loop->getSubLoops();
    for (auto BBIter = NewContextLoop.Loop->block_begin(),
              BBEnd = NewContextLoop.Loop->block_end();
         BBIter != BBEnd; ++BBIter) {
      auto BB = *BBIter;
      bool IsAtThisLevel = true;
      for (const auto &SubLoop : SubLoops) {
        if (SubLoop->contains(BB)) {
          IsAtThisLevel = false;
          break;
        }
      }
      if (!IsAtThisLevel) {
        continue;
      }

      // if (LoopUtils::getLoopId(NewContextLoop.Loop) ==
      //     "VerticalFilter::bb268") {
      //  LLVM_DEBUG(llvm::dbgs()
      //         << "Initialize memory accesses for loop VerticalFilter::bb268 "
      //         << BB->getName() << "\n");
      // }

      for (auto InstIter = BB->begin(), InstEnd = BB->end();
           InstIter != InstEnd; ++InstIter) {
        auto StaticInst = &*InstIter;
        if ((!llvm::isa<llvm::LoadInst>(StaticInst)) &&
            (!llvm::isa<llvm::StoreInst>(StaticInst))) {
          // This is not a memory access instruction.
          continue;
        }

        // if (LoopUtils::getLoopId(NewContextLoop.Loop) ==
        //     "VerticalFilter::bb268") {
        //  LLVM_DEBUG(llvm::dbgs()
        //         << "Initialize memory accesses for loop VerticalFilter::bb268
        //         "
        //         << Utils::formatLLVMInst(StaticInst) << "\n");
        // }
        Iter->second.insert(StaticInst);
      }
    }
  }

  for (auto StaticInst : Iter->second) {
    ContextInst CInst(CurrentContext, StaticInst);

    // if (Utils::formatLLVMInst(CInst.Inst) ==
    //     "_ZN3povL31All_CSG_Intersect_IntersectionsEPNS_13Object_StructEPNS_"
    //     "10Ray_StructEPNS_13istack_structE::bb86::tmp89(load)") {
    //  LLVM_DEBUG(llvm::dbgs() << "push loop level for to our target
    //   instruction\n");
    // }

    this->initializeStreamIfNecessary(LoopStack, CInst);
    this->activateStream(ActiveStreams, CInst);
  }
}

void InlineContextStreamPass::addAccess(const InlineContextPtr &CurrentContext,
                                        const std::list<ContextLoop> &LoopStack,
                                        ActiveStreamMap &ActiveStreams,
                                        DynamicInstruction *DynamicInst) {
  auto StaticInst = DynamicInst->getStaticInstruction();
  assert(StaticInst != nullptr && "Invalid llvm static instruction.");
  ContextInst CInst(CurrentContext, StaticInst);
  // if (Utils::formatLLVMInst(CInst.Inst) ==
  //     "_ZN3povL31All_CSG_Intersect_IntersectionsEPNS_13Object_StructEPNS_"
  //     "10Ray_StructEPNS_13istack_structE::bb86::tmp89(load)") {
  //  LLVM_DEBUG(llvm::dbgs() << "add access to our target instruction\n");
  // }
  this->initializeStreamIfNecessary(LoopStack, CInst);

  // Update the global count.
  { this->MemAccessInstCount.at(CInst)++; }

  this->activateStream(ActiveStreams, CInst);
  auto Iter = this->InstStreamMap.find(CInst);
  if (Iter == this->InstStreamMap.end()) {
    // This is possible as some times we have instructions not in a loop.
    return;
  }
  uint64_t Addr = 0;
  if (llvm::isa<llvm::LoadInst>(StaticInst)) {
    Addr = DynamicInst->DynamicOperands[0]->getAddr();
  } else {
    Addr = DynamicInst->DynamicOperands[1]->getAddr();
  }
  for (auto &Stream : Iter->second) {
    Stream.addAccess(Addr);
  }
}

void InlineContextStreamPass::endIter(const std::list<ContextLoop> &LoopStack,
                                      ActiveStreamMap &ActiveStreams) {
  assert(!LoopStack.empty() && "Empty loop stack when endIter is called.");
  const auto &EndedContextLoop = LoopStack.back();
  auto ActiveStreamIter = ActiveStreams.find(EndedContextLoop);
  if (ActiveStreamIter != ActiveStreams.end()) {
    for (auto &InstStream : ActiveStreamIter->second) {
      // Only call endIter for those streams at my level.
      auto &Stream = InstStream.second;
      assert(Stream->getContextLoop() == EndedContextLoop &&
             "Mismatch on ContextLoop for streams in ActiveStreams");
      // This will implicitly call the addMissingAccess.
      Stream->endIter();
    }
  }
}

void InlineContextStreamPass::popLoopStack(std::list<ContextLoop> &LoopStack,
                                           ActiveStreamMap &ActiveStreams) {
  // Poping from the loop stack automatically means that this iteration ends.
  assert(!LoopStack.empty() && "Empty loop stack when popLoopStack is called.");
  this->endIter(LoopStack, ActiveStreams);

  // Deactivate all the streams in this loop level.
  const auto &EndedContextLoop = LoopStack.back();
  auto ActiveStreamIter = ActiveStreams.find(EndedContextLoop);
  if (ActiveStreamIter != ActiveStreams.end()) {
    for (auto &ActiveInstStream : ActiveStreamIter->second) {
      ActiveInstStream.second->endStream();
    }
    ActiveStreams.erase(ActiveStreamIter);
  }

#ifdef DEBUG_TARGET_CLOOP
  if (EndedContextLoop.format() == DEBUG_TARGET_CLOOP) {
    LLVM_DEBUG(llvm::dbgs() << "Pop loop stack with context loop:\n"
                            << EndedContextLoop.beautify() << '\n');
  }
#endif

  // Pop from the loop stack.
  LoopStack.pop_back();
}

void InlineContextStreamPass::analyzeStream() {

  LLVM_DEBUG(llvm::dbgs() << "Stream: Start analysis.\n");

  std::list<ContextLoop> LoopStack;
  InlineContextPtr CurrentContext = InlineContext::getEmptyContext();
  ActiveStreamMap ActiveStreams;
  llvm::Instruction *PrevCallInst = nullptr;

  while (true) {

    auto NewInstIter = this->Trace->loadOneDynamicInst();

    llvm::Instruction *NewStaticInst = nullptr;
    llvm::Loop *NewLoop = nullptr;
    bool IsAtHeadOfCandidate = false;
    bool IsMemAccess = false;

    // Commit at 1000 instructions limits.
    while (this->Trace->DynamicInstructionList.size() > 1000) {
      this->Trace->commitOneDynamicInst();
    }

    if (NewInstIter != this->Trace->DynamicInstructionList.end()) {
      // This is a new instruction.
      NewStaticInst = (*NewInstIter)->getStaticInstruction();
      assert(NewStaticInst != nullptr && "Invalid static llvm instruction.");

      this->TracedDynInstCount.Val++;
      IsMemAccess = llvm::isa<llvm::LoadInst>(NewStaticInst) ||
                    llvm::isa<llvm::StoreInst>(NewStaticInst);

      auto LI = this->CachedLI->getLoopInfo(NewStaticInst->getFunction());
      NewLoop = LI->getLoopFor(NewStaticInst->getParent());

      if (NewLoop != nullptr) {
        IsAtHeadOfCandidate =
            LoopUtils::isStaticInstLoopHead(NewLoop, NewStaticInst);
      }
    } else {
      // This is the end.
      // We should end all the active streams.
      while (!LoopStack.empty()) {
        this->popLoopStack(LoopStack, ActiveStreams);
      }
      while (!this->Trace->DynamicInstructionList.empty()) {
        this->Trace->commitOneDynamicInst();
      }
      break;
    }

/**
 * ad-hoc debug section.
 */
#define DEBUG_RECURSIVE_DEPTH
#ifdef DEBUG_RECURSIVE_DEPTH

    {
      static size_t MaxDepth = 0;
      if (CurrentContext->size() > MaxDepth) {
        MaxDepth = CurrentContext->size();
        LLVM_DEBUG(llvm::dbgs() << "Maximum recursive depth updated to "
                                << MaxDepth << '\n');
        LLVM_DEBUG(llvm::dbgs() << "------- Context ---------\n");
        LLVM_DEBUG(llvm::dbgs() << CurrentContext->beautify() << '\n');
        LLVM_DEBUG(InlineContextStreamPass::DEBUG_LOOP_STACK(LoopStack));
      }
    }

#endif

// #define DEBUG_ACTIVE_STREAMS
#ifdef DEBUG_ACTIVE_STREAMS
    {
      static size_t BottomActiveStreams = 0;
      if (!LoopStack.empty()) {
        const auto &BottomCLoop = LoopStack.front();
        auto ActiveStreamIter = ActiveStreams.find(BottomCLoop);
        if (ActiveStreamIter != ActiveStreams.end()) {
          auto CurrentBottomActiveStreams = ActiveStreamIter->second.size();
          if (CurrentBottomActiveStreams > BottomActiveStreams) {
            BottomActiveStreams = CurrentBottomActiveStreams;
            LLVM_DEBUG(llvm::dbgs() << "Max bottom active streams updated to "
                                    << BottomActiveStreams << '\n');
            LLVM_DEBUG(llvm::dbgs() << "------- Context ---------\n");
            LLVM_DEBUG(llvm::dbgs() << CurrentContext->beautify() << '\n');
            LLVM_DEBUG(InlineContextStreamPass::DEBUG_LOOP_STACK(LoopStack));
            LLVM_DEBUG(InlineContextStreamPass::DEBUG_ACTIVE_STREAM_MAP(
                LoopStack, ActiveStreams));
          }
        }
      }
    }
#endif

    /**
     * Handle the previous call/invoke instruction by checking the datagraph's
     * DynamicFrameStack. The datagraph has the exact information as it can
     * see FunctionEnter message in the trace, however that is hidden from
     * higher transform pass, so we check it by looking at the dynamic frame
     * stack.
     */
    // LLVM_DEBUG(llvm::dbgs() << "Next inst is "
    //                    << Utils::formatLLVMInst(NewStaticInst) << '\n');
    if (PrevCallInst != nullptr) {

      /**
       * To detect if the new inst is actually from the callee.
       * 1. If the new inst is not ret, then we can just compare the size
       *    of the context and the frame stack.
       * 2. Otherwise, getting a ret means that the frame stack is already
       * poped, we can simply check if the ret and the previous call are from
       * the same function.
       */
      bool IsCalleeTraced = false;
      if (llvm::isa<llvm::ReturnInst>(NewStaticInst)) {
        IsCalleeTraced =
            NewStaticInst->getFunction() != PrevCallInst->getFunction();
      } else {
        auto ContextSize = CurrentContext->size();
        auto DynamicFrameStackSize = this->Trace->DynamicFrameStack.size();
        IsCalleeTraced = ContextSize + 2 == DynamicFrameStackSize;
      }

      // LLVM_DEBUG(llvm::dbgs() << "IsCalleeTraced " << IsCalleeTraced <<
      // '\n');

      if (IsCalleeTraced) {
        CurrentContext = CurrentContext->push(PrevCallInst);
        // LLVM_DEBUG(llvm::dbgs() << "Pushed context.\n");
        // LLVM_DEBUG(llvm::dbgs() << CurrentContext->beautify() << '\n');
        auto Callee = Utils::getCalledFunction(PrevCallInst);
        auto NewLoopStatus = INLINE_CONTINUOUS;
        if (Callee == nullptr) {
          // This is an indirect call or something.
          NewLoopStatus = INDIRECT_CONTINUOUS;
        }

        for (const auto &CLoop : LoopStack) {
          auto &CurrentStatus = this->LoopStatus.at(CLoop);
          CurrentStatus = InlineContextStreamPass::updateLoopStatusT(
              CurrentStatus, NewLoopStatus);
        }

        /**
         * If there is recursion, we have to mark all the context loops
         * recursive.
         */
        if (CurrentContext->isRecursive(NewStaticInst->getFunction())) {
          for (const auto &ContextLoop : LoopStack) {
            auto &CurrentStatus = this->LoopStatus.at(ContextLoop);
            // LLVM_DEBUG(llvm::dbgs() << "This is recursive.\n");
            CurrentStatus = InlineContextStreamPass::updateLoopStatusT(
                CurrentStatus, RECURSIVE);
          }
        }
      } else {
        // Untraced calls, we have to check if we support this function.
        auto Callee = Utils::getCalledFunction(PrevCallInst);
        if (Callee != nullptr &&
            (LoopUtils::LoopContinuityIgnoredFunctions.count(
                 Callee->getName().str()) ||
             Callee->isIntrinsic())) {
          // This is supported. We consider it as CONTINUOUS and do nothing.
        } else {
          // This is unsupported untraced call. We consider it as INCONTINUOUS.
          for (const auto &CLoop : LoopStack) {
            // LLVM_DEBUG(llvm::dbgs() << "Update for INCONTINUOUS\n");
            auto &CurrentStatus = this->LoopStatus.at(CLoop);
            // LLVM_DEBUG(llvm::dbgs() << "Get current status.\n");
            CurrentStatus = InlineContextStreamPass::updateLoopStatusT(
                CurrentStatus, INCONTINUOUS);
            // LLVM_DEBUG(llvm::dbgs()
            //       << "Update lop status with INCONTINUOUS for loop "
            //       << CLoop.format() << '\n');
          }
        }
      }
      PrevCallInst = nullptr;
    }

    // LLVM_DEBUG(llvm::dbgs() << "Done handling previous call inst.\n");

    ContextInst NewCInst(CurrentContext, NewStaticInst);

    /**
     * Pop the loop stack if it doesn't contain the new context inst.
     */
    while (!LoopStack.empty()) {
      if (LoopStack.back().contains(NewCInst)) {
        break;
      }
#ifdef DEBUG_TARGET_LOOP
      if (LoopUtils::getLoopId(LoopStack.back().Loop) == DEBUG_TARGET_LOOP) {
        LLVM_DEBUG(llvm::dbgs()
                   << "pop our target loop at inst "
                   << Utils::formatLLVMInst(NewStaticInst) << ".\n");
      }
#endif
      this->popLoopStack(LoopStack, ActiveStreams);
    }

    /**
     * Keep pop the loop stack if this is a ret instruction.
     * And also modify the InlineContext.
     */
    if (llvm::isa<llvm::ReturnInst>(NewStaticInst)) {
      if (!CurrentContext->empty()) {
        // Pop one context call site.
        CurrentContext = CurrentContext->pop();
        // LLVM_DEBUG(llvm::dbgs() << "Popped context.\n");
        // LLVM_DEBUG(llvm::dbgs() << CurrentContext->beautify() << '\n');
      }
    }

    // LLVM_DEBUG(llvm::dbgs() << "Done popping loop stack.\n");

    /**
     * Insert new loop if we are at the head.
     */
    if (NewLoop != nullptr && IsAtHeadOfCandidate) {
      ContextLoop NewCLoop(CurrentContext, NewLoop);
      if (LoopStack.empty() || LoopStack.back() != NewCLoop) {
        this->pushLoopStack(CurrentContext, LoopStack, ActiveStreams, NewCLoop);
      } else {
        // This means that we are at a new iteration.
        this->endIter(LoopStack, ActiveStreams);
      }
    }

    /**
     * If this is a call instruction, check if we can inline it.
     */
    if (Utils::isCallOrInvokeInst(NewStaticInst)) {
      // We do not know if the callee is actually traced.
      assert(PrevCallInst == nullptr &&
             "Unresolved previous call instruction.");
      auto Callee = Utils::getCalledFunction(NewStaticInst);
      if (Callee != nullptr && Callee->isIntrinsic()) {
        // Ignore intrinsic function calls.
      } else {
        PrevCallInst = NewStaticInst;
      }
    }

    /**
     * If this is a memory access, call addAccess to update the stream.
     */
    if (IsMemAccess) {
      this->DynMemInstCount.Val++;
      this->addAccess(CurrentContext, LoopStack, ActiveStreams, *NewInstIter);
      this->CacheWarmerPtr->addAccess(*NewInstIter, this->Trace->DataLayout);
    }
  }
}

} // namespace

#undef DEBUG_TYPE

char InlineContextStreamPass::ID = 0;
static llvm::RegisterPass<InlineContextStreamPass>
    X("inline-stream-pass", "Stream transform pass with inline", false, false);
