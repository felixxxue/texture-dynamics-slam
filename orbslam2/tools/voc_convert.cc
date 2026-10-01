#include "DBoW2/FORB.h"
#include "DBoW2/TemplatedVocabulary.h"
#include <iostream>
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
int main(int argc, char** argv){
  if(argc<3){ std::cerr<<"usage: voc_convert ORBvoc.txt ORBvoc.bin\n"; return 1; }
  ORBVocabulary v; if(!v.loadFromTextFile(argv[1])) return 2;
  v.saveToBinaryFile(argv[2]);
  ORBVocabulary w; if(!w.loadFromBinaryFile(argv[2])) return 3;
  std::cout << "text: words=" << v.size() << " levels=" << v.getDepthLevels() << " | bin: words=" << w.size() << std::endl;
  return 0; }
