#include "TestSetup.h"

#include <dirent.h>
#include <sys/stat.h>
#include <sys/types.h>

std::unique_ptr<llvm::Module>
makeLLVMModule(llvm::LLVMContext &Context, const std::string &ModuleFileName) {
  llvm::SMDiagnostic Err;
  return llvm::parseIRFile(ModuleFileName, Err, Context);
}

TestInput::TestInput(const std::string &_TestInputSource,
                     const std::string &_Pass)
    : TestInputSource(_TestInputSource), Pass(_Pass) {

  assert(this->TestInputSource.size() > 2 &&
         "Test input source file should be at least '.c'.");
  assert(this->TestInputSource.substr(this->TestInputSource.size() - 2) ==
             ".c" &&
         "Test input source file should have suffix '.c'.");

  this->Prefix =
      this->TestInputSource.substr(0, this->TestInputSource.size() - 2);
  this->BitCodeFile = this->Prefix + ".bc";
  this->InstUidMapFile = this->Prefix + ".inst.uid.txt";
  this->ProfileFile = this->Prefix + ".profile";
  this->TraceFile = this->Prefix + ".0.trace";
  this->OutputDataGraphFile = this->Prefix + "." + Pass + ".0.tdg";
  this->OutputDataGraphTextFile = this->OutputDataGraphFile + ".txt";
}

std::string TestInput::getAndCreateOutputDataGraphExtraFolder(
    const std::string &Pass) const {
  auto ExtraFolder = this->Prefix + "." + Pass + ".0.tdg.extra";

  if (!opendir(ExtraFolder.c_str())) {
    int Status =
        mkdir(ExtraFolder.c_str(), S_IRWXU | S_IRWXG | S_IROTH | S_IXOTH);
    assert(Status == 0 && "Failed to create the extra folder.");
  }

  return ExtraFolder;
}

void TestInput::setUpLLVMOptions(const std::string &Pass) const {

  /**
   * We need to manually set all the options.
   */
  DataGraphDetailLevel.reset();
  DataGraphDetailLevel.addOccurrence(0, "datagraph-detail", "standalone");
  assert(DataGraphDetailLevel.getNumOccurrences() == 1 &&
         "Should have one occurrence.");
  assert(DataGraphDetailLevel.getValue() ==
             DataGraph::DataGraphDetailLv::STANDALONE &&
         "Should be standalone mode.");

  GemForgeOutputDataGraphFileName.reset();
  GemForgeOutputDataGraphFileName.addOccurrence(0, "output-datagraph",
                                                this->getOutputDataGraphFile());

  GemForgeOutputExtraFolderPath.reset();
  GemForgeOutputExtraFolderPath.addOccurrence(
      0, "output-extra-folder-path",
      this->getAndCreateOutputDataGraphExtraFolder(Pass));

  GemForgeOutputDataGraphTextMode.reset();
  GemForgeOutputDataGraphTextMode.addOccurrence(0, "output-datagraph-text-mode",
                                                "true");

  TraceFileName.reset();
  TraceFileName.addOccurrence(0, "trace-file", this->getTraceFile());

  InstUIDFileName.reset();
  InstUIDFileName.addOccurrence(0, "gem-forge-inst-uid-file",
                                this->getInstUidMapFile());

  TraceFileFormat.reset();
  TraceFileFormat.addOccurrence(0, "trace-format", "protobuf");
}