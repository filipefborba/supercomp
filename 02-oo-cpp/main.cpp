#include "experiment.hpp"
#include "experimentLog.hpp"
#include "experimentPow.hpp"
#include "experimentPow3.hpp"
#include "experimentPow3Mult.hpp"
#include "experimentSum.hpp"

#include <iostream>

void do_experiment(Experiment &e, int n, std::string &name) {
    e.generate_vector(n);
    e.run();
    if (e.duration() < 0.1) {
        std::cout << name << double(e) << std::endl;
    }
}

void do_experiment_noprint(Experiment &e, int n) {
    e.generate_vector(n);
    e.run();
}

void min_max_experiment(Experiment &e, std::string &name) {
    double min;
    double max;
    std::cout << name << std::endl;
    for (int n = 10; n <= 1000000; n *=10) {
        std::cout << "n = " << n << std::endl;
        for (int i = 10; i > 0; i--) {
            do_experiment_noprint(e, n);
            if (i == 10) {
                min = e.duration();
                max = e.duration();
            } else {
                if (e.duration() > max) {
                    max = e.duration();
                }
                if (e.duration() < min) {
                    min = e.duration();
                }
            }
        }
        std::cout << "Min: " << min << std::endl;
        std::cout << "Max: " << max << std::endl << std::endl;
        min = 0;
        max = 0;
    }
}

int main() {
    int n = 1000000;

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

    std::cout << std::endl << "------Experiments--------" << std::endl;
    do_experiment(*log, n, s1);
    do_experiment(*pow, n, s2);
    do_experiment(*pow3, n, s3);
    do_experiment(*pow3mult, n, s4);
    do_experiment(*sum, n, s5);

    std::cout << std::endl << "------Min-&-Max--------" << std::endl;

    min_max_experiment(*log, s1);
    min_max_experiment(*pow, s2);
    min_max_experiment(*pow3, s3);
    min_max_experiment(*pow3mult, s4);
    min_max_experiment(*sum, s5);

    return 0;
}