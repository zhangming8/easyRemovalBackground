#include <thread>

#include "app.h"

namespace Algo
{
    void get_device_name(Args &args)
    {
        std::vector<std::string> devices = get_devices();
        std::cout << "[AITools] cuda device number: " << devices.size() << std::endl;
        std::string device_name = "";
        if (devices.empty())
        {
            std::cout << "[AITools] cannot find cuda device..." << std::endl;
            exit(-1);
        }
        else
        {
            int n = 1;
            for (auto &d : devices)
            {
                std::cout << "[AITools] total device: " << n << "/" << devices.size() << " name: " << d << std::endl;
                n++;
            }
            device_name = devices.at(0);
            std::cout << "[AITools] select first device: " << device_name << std::endl;
            replace_str(device_name, " ", "_");
            std::cout << "[AITools] new device name: " << device_name << std::endl;
        }

        if (device_name.empty())
        {
            std::cout << "[AITools] cannot find cuda device..." << std::endl;
            exit(-1);
        }
        args.device = device_name;
    }
}
