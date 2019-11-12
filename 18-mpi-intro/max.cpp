#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
#include <vector>
#include <cstdlib>
namespace mpi = boost::mpi;

int main(int argc, char *argv[])
{
    mpi::environment env(argc, argv);
    mpi::communicator world;
    int i = world.rank();
    int N = world.size();

    if (i == 0)
    {
        int max = 0;
        int recv_max;

        std::srand(0);
        int vec_size = 100000;
        std::vector<int> v(vec_size);
        std::generate(v.begin(), v.end(), std::rand);

        int chunk_size = vec_size / N;
        for (int j = 1; j < N; j++)
        {
            std::vector<int> v_slice(v.begin() + (j * chunk_size), v.begin() + ((1 + j) * chunk_size));
            world.send(j, 0, v_slice);
        }

        max = *std::max_element(v.begin(), v.begin() + chunk_size);

        for (int j = 1; j < N; j++)
        {
            world.recv(j, 0, recv_max);
            if (recv_max > max)
            {
                max = recv_max;
            }
        }

        std::cout << "Max element is " << max << std::endl;
    }
    else
    {
        std::vector<int> recv_v;
        world.recv(0, 0, recv_v);

        int result = *std::max_element(recv_v.begin(), recv_v.end());
        world.send(0, 0, result);
    }

    return 0;
}