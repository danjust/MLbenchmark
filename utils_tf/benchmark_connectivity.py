from utils_connectivity import benchmark_onedir


def benchmark_connectivity(
        devlist,
        precision,
        size_x,
        size_y,
        iterations):

    devlist = devlist.split(',')

    for host_dev in devlist:
        for remote_dev in devlist:
            timeUsed = benchmark_onedir.benchmark_onedir(
                    host_dev,
                    remote_dev,
                    precision,
                    size_x,
                    size_y,
                    iterations):
            print('%s --> %s: %.3f GBit / s' %(host_dev,remote_dev,size_x*size_y*precision/(timeUsed*1e9)))
