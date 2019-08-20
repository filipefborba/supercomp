#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <utility>
#include <vector>
#include <chrono>

class Experiment {
    private:
    ;
    public:
        Experiment();
        ~Experiment();

        int n; //size
        std::vector<double> arr;
        double dur; //duration of experiment in seconds

        std::vector<double> generate_vector(int n);
        double duration();
        std::pair<double, double> run(void);
        virtual void experiment_code(void);

        operator double();
        bool operator< (Experiment &e);
        bool operator< (double duration);
};

#endif