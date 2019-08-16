#ifndef EXPERIMENT_H
#define EXPERIMENT_H

#include <chrono>

class Experiment {
    private:
    ;
    public:
        Experiment();
        Experiment(int n);
        ~Experiment();

        int n; //size
        double* arr = new double[n]; //double array
        double dur; //duration of experiment in seconds

        double* generate_vector(int n);
        double duration();
        void run(void);
        virtual void experiment_code(void);

        operator double();
        bool operator< (Experiment &e);
        bool operator< (double duration);
};

#endif