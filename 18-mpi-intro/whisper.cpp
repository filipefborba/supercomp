#include <boost/mpi/environment.hpp>
#include <boost/mpi/communicator.hpp>
#include <iostream>
namespace mpi = boost::mpi;

void send_message(mpi::communicator world, int i, int N) {
    if (i == (N - 1)) {
        world.send(0, 0, i*i);
    } else {
        world.send(i+1, 0, i*i);
    }
}

void receive_message(mpi::communicator world, int i, int N) {
    int data;
    mpi::status msg;

    if (i == 0) {
        msg = world.recv(N - 1, 0, data);
    } else {
        msg = world.recv(i - 1, 0, data);
    }

    std::cout << "Recebido de " << msg.source() << ": " << data << ";\n";
}

int main(int argc, char* argv[]) {
    mpi::environment env(argc, argv);
    mpi::communicator world;

    int i = world.rank();
    int N = world.size();

    send_message(world, i, N);
    receive_message(world, i, N);

    return 0;
}