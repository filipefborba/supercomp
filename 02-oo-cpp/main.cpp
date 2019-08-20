#include "experiment.hpp"
#include "experimentLog.hpp"
#include "experimentPow.hpp"
#include "experimentPow3.hpp"
#include "experimentPow3Mult.hpp"
#include "experimentSum.hpp"

#include <iostream>
#include <utility>

std::pair<double, double> do_experiment(Experiment &e, int n) {
    e.generate_vector(n);
    return e.run();
}

void mean_std_experiment(Experiment &e, std::string &name) {
    std::vector<std::pair<double, double>> results_arr;
    std::cout << std::endl << name << std::endl;
    int n = 10; 
    int n_max = 1000000;

    // Run experiments
    for (n; n <= n_max; n *=10) {
        results_arr.push_back(do_experiment(e, n));
    }

    // Print results
    n = 10;
    for (int i = 0; i < results_arr.size(); i++) {
        std::cout << "n = " << n << std::endl;
        std::cout << "Mean: " << results_arr[i].first << " | Std Deviation: " << results_arr[i].second << std::endl;
        n *= 10;
    }
}

int main() {
    Experiment *log = new ExperimentLog();
    Experiment *pow = new ExperimentPow();
    Experiment *pow3 = new ExperimentPow3();
    Experiment *pow3mult = new ExperimentPow3Mult();
    Experiment *sum = new ExperimentSum();

    std::string s1 = "ExperimentLog: ";
    std::string s2 = "ExperimentPow: ";
    std::string s3 = "ExperimentPow3: ";
    std::string s4 = "ExperimentPow3Mult: ";
    std::string s5 = "ExperimentSum: ";

    std::cout << std::endl << "------Mean-&-Std--------" << std::endl;

    mean_std_experiment(*log, s1);
    mean_std_experiment(*pow, s2);
    mean_std_experiment(*pow3, s3);
    mean_std_experiment(*pow3mult, s4);
    mean_std_experiment(*sum, s5);

    return 0;
}