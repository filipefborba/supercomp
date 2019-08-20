#include <iostream>
#include <string>
#include <vector>

void print(std::vector<int> const &input)
{
    std::cout << "[";
	for (auto const& i: input) {
		std::cout << i << ", ";
	}
    std::cout << "]." << std::endl;
}

std::vector<int> find_all(std::string text, std::string term) {
    std::vector<int> ocurrences;

    std::size_t found = text.find(term);
    // Repete até o final da string (npos)
    while (found != std::string::npos) {
        ocurrences.push_back(found);
        found = text.find(term, found + term.size());
    }
    return ocurrences;
}

int main() {
    std::string line;
    std::getline(std::cin, line);
    std::string term = "hello";
    std::cout << "Found " << term << " in positions: ";
    print(find_all(line, term));
}