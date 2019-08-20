#include <iostream>
#include <memory>

int main() {
  std::unique_ptr<int> ptr = std::unique_ptr<int>(new int(0));
  for(int f=0;f<1024*1024*1024;f++) {
    ptr.reset(new int(f));
  }
  std::cout << "valor final = " << *ptr << std::endl;
}