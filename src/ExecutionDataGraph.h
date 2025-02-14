#ifndef LLVM_TDG_EXECUTION_DATA_GRAPH_H
#define LLVM_TDG_EXECUTION_DATA_GRAPH_H

#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"

#include "stream/StreamMessage.pb.h"

#include <functional>
#include <list>
#include <ostream>
#include <unordered_map>
#include <unordered_set>

// Represent an executionn data graph.
class ExecutionDataGraph {
public:
  using InstSet = std::unordered_set<const llvm::Instruction *>;
  using ValueList = std::list<const llvm::Value *>;

  ExecutionDataGraph(const llvm::Value *_ResultValue)
      : ResultValues({_ResultValue}) {}
  ExecutionDataGraph() {}
  ExecutionDataGraph(const ExecutionDataGraph &Other) = delete;
  ExecutionDataGraph(ExecutionDataGraph &&Other) = delete;
  ExecutionDataGraph &operator=(const ExecutionDataGraph &Other) = delete;
  ExecutionDataGraph &operator=(ExecutionDataGraph &&Other) = delete;

  virtual ~ExecutionDataGraph() {}

  const llvm::Value *getSingleResultValue() const {
    assert(this->ResultValues.size() == 1 && "No single result value");
    return this->ResultValues.back();
  }
  bool hasSingleResult() const { return this->ResultValues.size() == 1; }
  bool hasMultipleResult() const { return this->ResultValues.size() > 1; }
  const ValueList &getInputs() const { return this->Inputs; }
  const InstSet &getComputeInsts() const { return this->ComputeInsts; }
  bool hasCircle() const { return this->HasCircle; }

  /**
   * A hack to extend the result value with a tail Atomic instruction.
   * Can be AtomicRMW or AtomicCmpXchg. Then the DataGraph will generate two
   * functions, one for loaded value and one for stored value.
   * e.g. for atomiccmpxchg stream, the __store() function returns the final
   * value stored to the place, and __load() function returns the value for
   * the core.
   *
   * The Atomic instruction is transformed to its normal version, with
   * the input as the last function argument and result as returned value.
   */
  void extendTailAtomicInst(
      const llvm::Instruction *AtomicInst,
      const std::vector<const llvm::Instruction *> &FusedLoadOps);
  const llvm::Instruction *getTailAtomicInst() const {
    return this->TailAtomicInst;
  }
  bool hasTailAtomicInst() const { return this->TailAtomicInst != nullptr; }

  llvm::Type *getReturnType(bool IsLoad) const;

  /**
   * Generate a function takes the input and returns the value.
   * @param IsLoad: Only useful when there is a tail atomic.
   * Returns the inserted function.
   */
  llvm::Function *generateFunction(const std::string &FuncName,
                                   std::unique_ptr<llvm::Module> &Module,
                                   bool IsLoad = false) const;

  /**
   * Check if this function can be represented as single compute op.
   */
  ::LLVM::TDG::ExecFuncInfo::ComputeOp getComputeOp() const;

protected:
  std::list<const llvm::Value *> ResultValues;
  std::list<const llvm::Value *> Inputs;
  std::unordered_set<const llvm::Value *> ConstantValues;
  InstSet ComputeInsts;
  const llvm::Instruction *TailAtomicInst = nullptr;
  std::vector<const llvm::Instruction *> FusedLoadOps;
  bool HasCircle = false;

  using DFSTaskT = std::function<void(const llvm::Instruction *)>;
  void dfsOnComputeInsts(const llvm::Instruction *Inst, DFSTaskT Task) const;

  llvm::FunctionType *createFunctionType(llvm::Module *Module,
                                         bool IsLoad) const;

  using ValueMapT = std::unordered_map<const llvm::Value *, llvm::Value *>;
  void translate(std::unique_ptr<llvm::Module> &Module,
                 llvm::IRBuilder<> &Builder, ValueMapT &ValueMap,
                 const llvm::Instruction *Inst) const;

  bool detectCircle() const;

  llvm::Value *generateTailAtomicRMW(llvm::IRBuilder<> &Builder,
                                     llvm::Value *AtomicArg,
                                     llvm::Value *RhsArg, bool IsLoad) const;
  llvm::Value *generateTailAtomicCmpXchg(llvm::IRBuilder<> &Builder,
                                         llvm::Value *AtomicArg,
                                         llvm::Value *CmpValue,
                                         llvm::Value *XchgValue,
                                         bool IsLoad) const;
};

#endif