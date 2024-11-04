import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Utility program to compress and decompress trajectory files.")
    subparsers = parser.add_subparsers(help = "subcommand help", dest="mode", required=True)
    parser_compress_parent = argparse.ArgumentParser(add_help=False)
    parser_decompress = subparsers.add_parser("decompress", help="Decompress a file.")

    parser_compress_parent.add_argument("traj_file", metavar="in_file", type=str, help="The name of the trajectory file which should be compressed.")
    parser_compress_parent.add_argument("out_file", metavar="out_file", type=str, help="The name of the compressed output file. The extension '.zip' will be appended")
    parser_compress_parent.add_argument("-m", "--method", dest="method", choices=["jpeg", "numpy", "numpy16bit", "numpy8bit", "sz3"], type=str, default="jpeg",
                        help="""The method used to compress the data, the options are:\n
                        'jpeg' compress images using jpeg compression.\n
                        'numpy' Store in numpy file, no compression will take place.\n
                        'numpy16bit' Store in numpy file after converting to 16 bit integers, compression will take place because of quantization.\n
                        'numpy8bit' Store in numpy file after converting to 8 bit integers, compression will take place because of qunatization.\n
                        'sz3' compress using the sz3 compression algorithm.
                        The default value is jpeg""")
    parser_compress_parent.add_argument("-r", "--crf", dest="crf", type=int, default=10, help="Set the compression factor you would like to have. Default 10.")
    parser_compress_parent.add_argument("-f", "--factor", dest="fac", type=float, default=1.5, help="Set the factor of numer of pixels compared to trajectories, should be larger than 1.5. setting it to 1.5 means that we have 1.5 times more pixels than trajectories. Default 1.5.")
    parser_compress_parent.add_argument("-c", "--config", metavar="file", dest="config", type=str, help="Load the compression methods and ratios from an external yaml config file where they are specified for each data variable.")
    parser_compress_parent.add_argument("-e", "--exclude", metavar="vars", dest="exclude", type=str, nargs="+", default=[], help="Data variables which should not be compressed.")
    parser_compress_parent.add_argument("-n", "--num_workers", metavar="N", dest="num_workers", type=int, default=1, help="The number of parallel workers to be used during compresssion, optimally matches the number of pressure levels.")
    parser_compress_parent.add_argument("--color_method", metavar="color_method", dest="color_method", type=str, default="xyz", choices=["none", "circle", "luigi", "hsv", "xyz"], help="The method to use for color encoding. Default xyz.")
    parser_compress_parent.add_argument("--color_bits", metavar="bits", dest="color_bits", type=int, default=16, choices=[8,16], help="The number of bits that should be used for each color channel. Default 16.")
    parser_compress_parent.add_argument("--delta_method", metavar="method", dest="delta_method", type=str, default="r1", choices=["none", "n1", "r1"], help="Set the delta method to be used for compression. Default r1.")
    parser_compress_parent.add_argument("-b", "--bin", metavar="N", dest="bin", type=int, default=0, help="If the trajectories are not on distinct pressure levels at the first time step this can be used to force them to be binned into N bins, if N=0 then it is assumed, that the trajectories are on distinct pressure levels. Default 0 (distinct pressure levels are assumed).")
    parser_compress_parent.add_argument("--mapping_func", metavar="func", dest="mapping_func", type=str, default="bipar", choices=["bipar", "lp"], help="Select the method to be used for mapping, always chosse 'bipar'")

    parser_decompress.add_argument("out_file", metavar="in_file", type=str, help="The file which should be decompressed.")
    parser_decompress.add_argument("uncom_file", metavar="out_file", type=str, help="The file in which this should be decompressed into.")

    parser_compress = subparsers.add_parser("compress", help="Compress a file", parents=[parser_compress_parent])
    parser_test = subparsers.add_parser("test", help="Run a test where a file is compressed and the decompressed again with some test run", parents=[parser_compress_parent])


    args = parser.parse_args()


    import xarray as xr
    import numpy as np
    import os
    import time
    import json
    import pickle
    import yaml
    import copy
    from tabulate import tabulate
    from psit import Psit
    import math_helper as ap


    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, np.ndarray):
                return f"numpy.ndarray of shape {obj.shape}"
            if isinstance(obj, np.float32):
                return float(obj)
            return json.JSONEncoder.default(self, obj)


    def compress(args):
        t_file = xr.open_dataset(args.traj_file)
        compressor = Psit()
        start_compress = time.time()
        compressor.compress(t_file, args.out_file, copy.deepcopy(args.crf), args.exclude, copy.deepcopy(args.method), args.color_method, args.color_bits, copy.deepcopy(args.delta_method), args.bin, args.mapping_func, args.num_workers, args.fac)
        end_compress = time.time()

        orig_size = os.path.getsize(args.traj_file)
        if args.out_file.endswith(".zip"):
            com_size = os.path.getsize(f"{args.out_file}")
        else:
            com_size = os.path.getsize(f"{args.out_file}.zip")

        print("================================================================================")
        print("                                  Configuration                                 ")
        print("================================================================================")
        print("")
        data = []
        for d in args.__dict__:
            data.append([d, args.__dict__[d]])
        print(tabulate(data, headers=["parameter", "value"]))
        print("")
        print("")
        print("================================================================================")
        print("                             Compression Performance                            ")
        print("================================================================================")
        print("")
        print(f"Original size:      {orig_size/1000000}MB")
        print(f"Compressed size:    {com_size/1000000}MB")
        print(f"Compression factor: {orig_size/com_size}")
        print("")
        print("")
        print("================================================================================")
        print("                                     Runtime                                    ")
        print("================================================================================")
        print("")
        print(f"Compression:   {end_compress - start_compress}s")
        print("")
        print("")


    def decompress(args):
        compressor = Psit()
        start_decompress = time.time()
        uncompressed_data = compressor.decompress(args.out_file)
        uncompressed_data.to_netcdf(args.uncom_file)
        end_decompress = time.time()

        uncomp_size = os.path.getsize(args.uncom_file)
        if args.out_file.endswith(".zip"):
            com_size = os.path.getsize(f"{args.out_file}")
        else:
            com_size = os.path.getsize(f"{args.out_file}.zip")
        print("================================================================================")
        print("                             Compression Performance                            ")
        print("================================================================================")
        print("")
        print(f"Uncompressed size:      {uncomp_size/1000000}MB")
        print(f"Compressed size:    {com_size/1000000}MB")
        print(f"Compression factor: {uncomp_size/com_size}")
        print("")
        print("")
        print("================================================================================")
        print("                                     Runtime                                    ")
        print("================================================================================")
        print("")
        print(f"Decompression:   {end_decompress - start_decompress}s")
        print("")
        print("")
        

    def test(args):
        t_file = xr.open_dataset(args.traj_file)
        compressor = Psit()
        start_compress = time.time()
        compressor.compress(t_file, args.out_file, copy.deepcopy(args.crf), args.exclude, copy.deepcopy(args.method), args.color_method, args.color_bits, copy.deepcopy(args.delta_method), args.bin, args.mapping_func, args.num_workers, args.fac)
        end_compress = time.time()
        start_decompress = time.time()
        u_file = compressor.decompress(args.out_file)
        end_decompress = time.time()
        
        o_data = dict()
        u_data = dict()
        for d in t_file.data_vars:
            if d not in args.exclude:
                o_data[d] = t_file[d].values.squeeze()
                u_data[d] = u_file[d].values.squeeze()
        orig_size = os.path.getsize(args.traj_file)
        com_size = os.path.getsize(f"{args.out_file}.zip")

        print("================================================================================")
        print("                                  Configuration                                 ")
        print("================================================================================")
        print("")
        data = []
        for d in args.__dict__:
            data.append([d, args.__dict__[d]])
        print(tabulate(data, headers=["parameter", "value"]))
        print("")
        print("")
        print("================================================================================")
        print("                             Compression Performance                            ")
        print("================================================================================")
        print("")
        print(f"Original size:      {orig_size/1000000}MB")
        print(f"Compressed size:    {com_size/1000000}MB")
        print(f"Compression factor: {orig_size/com_size}")
        print("")
        print("")
        print("================================================================================")
        print("                                     Runtime                                    ")
        print("================================================================================")
        print("")
        print(f"Compression:   {end_compress - start_compress}s")
        print(f"Decompression: {end_decompress - start_decompress}s")
        print("")
        print("")
        print("================================================================================")
        print("                            Compression-error summary                           ")
        print("================================================================================")
        print("All the units of the errors are of the same unit as the original data.")
        er_data = ap.pretty_print(o_data, u_data, args.color_method == "xyz")
        ap.er_over_time(o_data, u_data, args.color_method == "xyz")

        out_dict = dict()
        out_dict["configuration"] = args.__dict__
        out_dict["compression_performance"] = {"orig_size":orig_size, "compressed_size" : com_size, "ratio" : orig_size/com_size}
        out_dict["runtime"] = {"compression" : end_compress - start_compress, "decompression" : end_decompress - start_decompress}
        out_dict["error_data"] = er_data
        out_dict["differences"] = ap.calc_dif(o_data, u_data, args.color_method == "xyz")
        print("================================================================================")
        print("                                      Output                                    ")
        print("================================================================================")
        print(f"The following dictionary will be saved as a pickle file into {args.out_file}_data.pick")
        print(json.dumps(out_dict, sort_keys=True, indent=2, cls=NumpyEncoder))
        f = open(f"{args.out_file}_data.pick", "wb")
        pickle.dump(out_dict, f)
        f.close()




    if (args.mode == "compress" or args.mode == "test") and args.config is not None:
        f = open(args.config, "r")
        try:
            conf = yaml.safe_load(f)
            if "method" in conf:
                args.method = conf["method"]
            if "crf" in conf:
                args.crf = conf["crf"]
            if "factor" in conf:
                args.factor = conf["factor"]
            if "num_workers" in conf:
                args.num_workers = conf["num_workers"]
            if "color_method" in conf:
                args.color_method = conf["color_method"]
            if "color_bits" in conf:
                args.color_bits = conf["color_bits"]
            if "delta_method" in conf:
                args.delta_method = conf["delta_method"]
            if "exclude" in conf:
                args.exclude = conf["exclude"]
        except yaml.YAMLError as exc:
            print(exc)
            exit(-1)
        f.close()


    match args.mode:
        case "compress":
            compress(args)
        case "decompress":
            decompress(args)
        case "test":
            test(args)


