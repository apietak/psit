import os
import numpy as np
import scipy.sparse
import constriction
import glymur
import pickle
import math
from scipy.spatial import cKDTree
import shutil
from tqdm import tqdm
import multiprocessing as mp
import scipy
import sys
import xarray as xr
from scipy.interpolate import RBFInterpolator
from enum import IntEnum
import gurobipy as gp
from gurobipy import GRB
from datetime import datetime

sys.path.append(f"{os.path.dirname(os.path.realpath(__file__))}/SZ3/tools/pysz")
from pysz import SZ


import find_idx_wrapper as ctt


class PressureMeta:
    def __init__(self, pressure, nt, nv, n):
        self.pressure = pressure
        self.nt = nt
        self.nv = nv
        self.n = n
        self.mapping = np.empty(0)
        self.jpeg_metas = []

class Psit:
    class CompressionMethod(IntEnum):
        NUMPY=1
        NUMPY16=2
        NUMPY8=3
        JPEG=4
        SZ3=5
    
    class ColorMethod(IntEnum):
        NONE=1
        CIRCLE=2
        LUIGI=3
        HSV=4
        XYZ=5

    class DeltaMethod(IntEnum):
        NONE=1,
        N1=2,
        R1=3
    
    class ParallelWorker:
        def __init__(self, data, function, level_mapping, config):
            self.data = data
            self.function = function
            self.level_mapping = level_mapping
            self.config = config

        def work(self, level_idx):
            idxs = np.where(self.level_mapping == level_idx)[0].astype(np.int32)
            cur_data = {}
            for d in self.data:
                cur_data[d] = self.data[d][:,idxs]
            meta = self.function(cur_data, level_idx, self.config)
            meta.mapping = np.copy(idxs)
            return meta

    class Skeleton:
        def __init__(self):
            self.dims = dict()
            self.attrs = dict()
            self.shape = dict()
            self.dtype = dict()

            self.global_attrs = dict()
            self.coords = dict()
            self.ballast = dict()


    def __init__(self) -> None:
        data = np.loadtxt(f"{os.path.dirname(os.path.realpath(__file__))}/coords_small.csv", delimiter=";")
        data = data.astype(np.int32)
        data[:,1] = -1.*data[:,1]
        data = data/7.
        self.snake_data = data

    def _get_method(self, method: str) -> CompressionMethod:
        match method:
            case "jpeg":
                return self.CompressionMethod.JPEG
            case "numpy":
                return self.CompressionMethod.NUMPY
            case "numpy16bit":
                return self.CompressionMethod.NUMPY16
            case "numpy8bit":
                return self.CompressionMethod.NUMPY8
            case "sz3":
                return self.CompressionMethod.SZ3
            case _:
                print(f"ERROR: Compression methodn '{method}' not found exiting.")
                exit(-1)


    def _get_color_method_and_type(self, color_method, color_bits):
        color_type = np.uint8
        if color_bits == 8:
            color_type = np.uint8
        elif color_bits == 16:
            color_type = np.uint16
        else:
            print(f"ERROR: Invalid number of color bits '{color_bits}', supported numbers are 8 and 16.")
            exit(-1)

        match color_method:
            case "none":
                color_method_c = self.ColorMethod.NONE
            case "circle":
                color_method_c = self.ColorMethod.CIRCLE
            case "luigi":
                color_method_c = self.ColorMethod.LUIGI
            case "hsv":
                color_method_c = self.ColorMethod.HSV
            case "xyz":
                color_method_c = self.ColorMethod.XYZ
            case _:
                print(f"ERROR: Invalid color encoding type '{color_method}', exiting.")
                exit(-1)
        
        return (color_method_c, color_type)
    
    def _get_delta_method(self, delta_method) -> DeltaMethod:
        match delta_method:
            case "none":
                return self.DeltaMethod.NONE
            case "n1":
                return self.DeltaMethod.N1
            case "r1":
                return self.DeltaMethod.R1

    
    def compress(self, dataset: xr.Dataset, filename: str, crf: int|dict, exclude: list = [], method: str|dict = "jpeg", color_method: str = "xyz", color_bits: int = 16, delta_method: str|dict = "r1", bin: int = 0, mapping_func: str = "bipar", num_workers: int = 1, factor: float = 1.5):
        """
        Compress trajectory data into file.
        
        Args:
            dataset:     An xarray Dataset containing all of the information.
            filename:    The filename to which one wants to write the data.
            crf:         Compression factor, i.e. the amount of compression that is desired. Either a single integer or a dictionary from data variable name to integers depending on if one wants to have a different compression factor per data variable.
            exclude:     List with all the data variable names which should not be compressed. E.g. 'BASEDATE'.
            method:      The method used for the compression, avaiable methods are "numpy", "numpy16bit", "numpy8bit", "jpeg", "sz3".
                         This is either a single string or a dictionary from data variable names to strings depending on if one wants to define
                         a compression method per data variable or one for all.
            color_method: The color method to be used for compression of the longitude data variable. Can be "none", "circle", "luigi", "hsv".
            color_bits:  The amount of bits to use for each color channel when using colour compression. Either 8 or 16.
            delta_method: The delta encoding method to use, valid methods are "none", "n1", "r1". "n1" is the method which operates on the perfect grids, while "r1" uses the reconstructed ones. Can either be a string or a dictionary from data variables to strings depending on if one wants to set a delta method for each data varaible individually or for all them at the same time. "r1" is the recomended option.
            bin:         Can be used to conrol the number of pressure levels the trajectories should be binned into in the first time step, if no binning is required set this to 0. 
            mapping_func:     A string telling which mapping method to take, either "bipar" or "lp", always choose "bipar".
            num_workers: Number of parallel workers, should be between 1 and number of levels
            factor:      The ratio between number of pixels and number of trajectories. factor=1.5 means that we have 1.5 times more pixels than trajectories.
        Returns:
            Creates a file called "<filename>.zip" in which the compressed data is stored.
        """
        
        # Set up the directory
        if filename.endswith(".zip"):
            folder = filename[:-4]
        else:
            folder = f"{filename}"
        image_folder = "images"
        if os.path.exists(f"{filename}.zip"):
            os.remove(f"{filename}.zip")
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
        os.makedirs(os.path.join(folder,image_folder))


        # From the dataset extract the actual data to compress and create a skeleton for the rest
        skeleton = self.Skeleton()
        skeleton.global_attrs = dataset.attrs
        # add note that compressed with psit to the history
        skeleton.global_attrs["history"] = f"{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}: Compressed with psit{(';' + skeleton.global_attrs['history']) if 'history' in skeleton.global_attrs else ''}"
        skeleton.coords = dataset.coords.to_dataset().coords
        data = dict()
        for i in dataset.data_vars:
            if i not in exclude:
                data[i] = dataset[i].values.squeeze()
                skeleton.attrs[i] = dataset[i].attrs
                skeleton.dims[i] = dataset[i].dims
                skeleton.shape[i] = dataset[i].values.shape
                skeleton.dtype[i] = dataset[i].dtype
            else:
                skeleton.ballast[i] = dataset[i]

        # check that all data variables to compress have same dimensions
        shape = None
        for d in data:
            if shape == None:
                shape = data[d].shape
            if data[d].shape != shape:
                print("ERROR: Not all data variables to compress have same dimensions, this is not supported. Add data variables which whould not be compressed to the exclude list.")
                exit(-1)


        # setup the method
        if type(method) is not dict:
            tmp = self._get_method(method)
            method = dict()
            for d in data:
                method[d] = tmp
        else:
            if data.keys() != method.keys():
                print("ERROR: The method dictionary does not contain the same data variables as the data to compress.")
                print("data variables")
                print(data.keys())
                print("methods")
                print(method.keys())
                exit(-1)
            for d in method:
                method[d] = self._get_method(method[d])
        
        
        # setup the compression ratios
        if type(crf) is not dict:
            tmp = crf
            crf = dict()
            for d in data:
                crf[d] = tmp
        else:
            if data.keys() != crf.keys():
                print("ERROR: The crf dictionary does not contain the same data variables as the data to compress.")
                print("data variables")
                print(data.keys())
                print("ratios")
                print(crf.keys())
                exit(-1)
        
        # setup the color encoding method
        color_method, color_type = self._get_color_method_and_type(color_method, color_bits)


        # setup the delta encoding method
        if type(delta_method) is not dict:
            tmp = self._get_delta_method(delta_method)
            delta_method = dict()
            for d in data:
                delta_method[d] = tmp
        else:
            if data.keys() != delta_method.keys():
                print("ERROR: The delta_method dictionary does not contain the same data variables as the data to compress.")
                print("data variables")
                print(data.keys())
                print("delta_methods")
                print(delta_method.keys())
                exit(-1)
            for d in delta_method:
                delta_method[d] = self._get_delta_method(delta_method[d])


        # Save the skeleton to disk for later usage
        f = open(os.path.join(folder,"skeleton.pick"), "wb")
        pickle.dump(skeleton, f)
        f.close()


        # extract the pressure levels
        if bin == 0:
            pressure_levels, indices = np.unique(data["p"][0,:], return_inverse=True)
            #pressure_levels = np.unique(data["p"][0,:])
            if (pressure_levels.shape[0] > 40):
                print("WARNING: more that 40 pressure levels found, data might not be split into distinct levels.")
            level_count = len(pressure_levels)
        else:
            min_p = data["p"][1,:].min()
            max_p = data["p"][1,:].max()
            num_traj = data["p"].shape[1]
            time = 1
            sep = (num_traj/float(bin))*np.arange(1,bin+1)
            indices = sep.searchsorted(np.arange(num_traj))
            indices = indices[data["p"][time,:].argsort().argsort()]
            #bins = np.linspace(min_p,max_p, bin)
            #indices = np.digitize(data["p"][1,:], bins)-1
            hist = np.bincount(indices)
            level_count = bin
        
        if mapping_func == "bipar":
            mapping_func = self._create_mapping_file
        elif mapping_func == "lp":
            mapping_func = self._create_mapping_file_lp
        else:
            print(f"ERROR: Unknown mapping function '{mapping_func}' used.")
            exit(-1)

        config = {
            "folder" : folder,
            "image_folder" : image_folder,
            "method" : method,
            "crf" : crf,
            "color_method" : color_method,
            "color_type" : color_type,
            "mapping_func" : mapping_func,
            "delta_method" : delta_method,
            "factor" : factor
            }
        # Setup the parallel workers
        metas = []
        worker = self.ParallelWorker(data, self._compress_level, indices, config)
        # Run the compression of the different levels
        print("Starting workers...")
        with mp.Pool(num_workers) as pool:
            metas = pool.map(worker.work, np.arange(0,level_count, 1, dtype=np.int32))
        print("Workers finished.")
        # Order the meta files according to pressure level
        metas_ordered = []
        for p in np.arange(0,level_count, 1, dtype=np.int32):
            for j in metas:
                if j.pressure == p:
                    j.mapping = self._compress_idxs(j.mapping)
                    metas_ordered.append(j)
                    break

        # Write the meta file to disk
        num_traj = data["p"].shape[1]
        meta_content = (num_traj, metas_ordered)
        meta_file = open(os.path.join(folder, "meta.pick"), "wb+")
        pickle.dump(meta_content, meta_file)
        meta_file.close()
        
        # create the finished archive
        shutil.make_archive(folder, 'zip', folder)
        shutil.rmtree(folder)
    
    def decompress(self, filename: str):
        """
        Decompress a compressed trajectory file into a dictionary.

        Args:
            filename: The name of the compressed trajectory file
        Returns:
            An xarray Dataset wich contains all the trajectory data.
        """
        # the folder in which this is stored
        if filename.endswith(".zip"):
            folder = filename[:-4]
        else:
            folder = filename
        
        # first unpack the compressed zip file
        shutil.unpack_archive(f"{folder}.zip", folder)

        # read the meta information
        meta_list_file = open(os.path.join(folder, "meta.pick"), "rb")
        num_traj, metas = pickle.load(meta_list_file)
        meta_list_file.close()

        # Read the skeleton from disk
        f = open(os.path.join(folder, "skeleton.pick"), "rb")
        skeleton = pickle.load(f)
        f.close()



        # main loop over all the meta files
        data = dict()
        for meta in tqdm(metas, desc="decompression"):
            local_data = dict()
            mapping = self._create_inverse_mapping_file(os.path.join(folder, f"mapping_{meta.pressure}.npy"))
            last_img_data = None
            for res in meta.jpeg_metas:
                # read data from meta file
                var = res["data_variable"]
                time = res["time"]
                method = res["method"]
                color_method = res["color_method"]
                color_type = res["color_type"]
                delta_method = res["delta_method"]
                match method:
                    case self.CompressionMethod.JPEG:
                        use_hue = var == "lon" and color_method != self.ColorMethod.NONE
                        img_data = self._jp2_decoding(res, folder, use_hue, color_method, color_type, self.DeltaMethod.NONE if time == 0 else delta_method, last_img_data)
                        if use_hue and delta_method != self.DeltaMethod.NONE:
                            last_img_data = np.copy(img_data)
                    case self.CompressionMethod.NUMPY:
                        img_data = np.load(os.path.join(folder,res["jp2_path"]))
                    case self.CompressionMethod.NUMPY16:
                        img_data = np.load(os.path.join(folder, res["jp2_path"]))
                        type = np.uint16
                        level_max = res["level_max"]
                        level_min = res["level_min"]
                        maxval = np.iinfo(type).max
                        img_data = img_data.astype(np.float32) / maxval
                        img_data = img_data * (level_max - level_min) + level_min
                    case self.CompressionMethod.NUMPY8:
                        img_data = np.load(os.path.join(folder,res["jp2_path"]))
                        type = np.uint8
                        level_max = res["level_max"]
                        level_min = res["level_min"]
                        maxval = np.iinfo(type).max
                        img_data = img_data.astype(np.float32) / maxval
                        img_data = img_data * (level_max - level_min) + level_min
                    case self.CompressionMethod.SZ3:
                        img_data = self._sz3_decoding(res, folder)
                    case _:
                        print("ERROR: Unknown compression method.")
                        exit(-1)
                if delta_method != self.DeltaMethod.NONE and not (method == self.CompressionMethod.JPEG and var == "lon" and color_method != self.ColorMethod.NONE):
                    if time == 0:
                        last_img_data = np.copy(img_data)
                    else:
                        img_data = last_img_data + img_data
                        last_img_data = np.copy(img_data)
                if color_method == self.ColorMethod.XYZ and var == "lon":
                    lon_data = img_data[0,mapping[:,0], mapping[:,1]]
                    lat_data = img_data[1,mapping[:,0], mapping[:,1]]
                    if "lon" not in local_data:
                        local_data["lon"] = np.empty((meta.nt,meta.n))
                        local_data["lat"] = np.empty((meta.nt,meta.n))
                    local_data["lon"][time] = lon_data
                    local_data["lat"][time] = lat_data
                else:
                    cur_data = img_data[mapping[:,0], mapping[:,1]]
                    if var not in local_data:
                        local_data[var] = np.empty((meta.nt,meta.n))
                    local_data[var][time] = cur_data

            for d in local_data:
                if d not in data:
                    data[d] = np.zeros((local_data[d].shape[0], num_traj))
                for t in range(local_data[d].shape[0]):
                    mapping_idxs = self._decopmress_idxs(meta.mapping)
                    data[d][t,mapping_idxs] = local_data[d][t,:]
        shutil.rmtree(folder)


        # Reconstruct the Dataset from the data and the skeleton
        data_vars = dict()
        for d in data:
            data[d].shape = skeleton.shape[d]
            data[d] = data[d].astype(skeleton.dtype[d])
            cur = xr.DataArray(data = data[d], dims=skeleton.dims[d], name=d, attrs=skeleton.attrs[d])
            data_vars[d] = cur
        data_vars = data_vars | skeleton.ballast
        skeleton.global_attrs["history"] = f"{datetime.now().strftime('%a %b %d %H:%M:%S %Y')}: Decompressed with psit; {skeleton.global_attrs['history']}"
        out = xr.Dataset(data_vars=data_vars, coords=skeleton.coords, attrs=skeleton.global_attrs)
        return out
    
    
    def _compress_level(self, data: dict, pressure, config):
        n = data[next(iter(data))].shape[1]
        nt = data[next(iter(data))].shape[0]
        nv = len(data)

        meta = PressureMeta(pressure, nt, nv, n)

        # Figure out the image dimensions
        n = data[next(iter(data))].shape[1]
        y_pixels = int(math.ceil(math.sqrt(config["factor"]*n/2.)))
        x_pixels = 2*y_pixels


        #mapping = self._create_mapping_file(os.path.join(folder,f"mapping_{pressure}.npy"), x_pixels, y_pixels, np.median(data["lon"][:,:],axis=0), np.median(data["lat"][:,:],axis=0))
        #mapping = self._create_mapping_file(os.path.join(folder,f"mapping_{pressure}.npy"), x_pixels, y_pixels, data["lon"][6,:], data["lat"][6,:])
        mapping, mask = config["mapping_func"](os.path.join(config["folder"],f"mapping_{pressure}.npy"), x_pixels, y_pixels, data["lon"][0,:], data["lat"][0,:])
        fullness = np.zeros(n)
        fullness[mapping.flatten()] = 1.
        assert(len(fullness.nonzero()[0]) == n)
        for d in data:
            if d == "lat" and config["color_method"] == self.ColorMethod.XYZ:
                continue

            cur_method = config["method"][d]
            cur_ratio = config["crf"][d]
            cur_delta_method = config["delta_method"][d]
            last_pixel_data = None
            for t in range(nt):
                cur_data = data[d][t,:]
                name = f"{d}_{pressure}_{t}"
                pixel_data = cur_data[mapping.flatten()]
                pixel_data.shape = mapping.shape
                if d == "lon" and config["color_method"] == self.ColorMethod.XYZ:
                    lat_data = data["lat"][t,:][mapping.flatten()]
                    lat_data.shape = mapping.shape
                    pixel_data = np.stack((pixel_data, lat_data))
                if cur_delta_method != self.DeltaMethod.NONE and not (cur_method == self.CompressionMethod.JPEG and d == "lon" and config["color_method"] != self.ColorMethod.NONE):
                    if t == 0:
                        last_pixel_data = np.copy(pixel_data)
                    else:
                        if cur_delta_method == self.DeltaMethod.N1:
                            tmp = np.copy(pixel_data)
                            pixel_data = np.copy(pixel_data - last_pixel_data)
                            last_pixel_data = tmp

                #print("start_interp")
                #rbf = RBFInterpolator(np.array((mask_rows, mask_cols)).T, pixel_data[mask==1], kernel='linear', neighbors=3)
                #interpolated_values = rbf(np.array((mask_nan_rows, mask_nan_cols)).T)
                #pixel_data[mask==0] = interpolated_values
                #print("end_interp")
                res = dict()
                match cur_method:
                    case self.CompressionMethod.JPEG:
                        use_hue = d == "lon" and config["color_method"] != self.ColorMethod.NONE
                        if cur_delta_method != self.DeltaMethod.NONE and use_hue and t == 0:
                            last_pixel_data = np.zeros_like(pixel_data)
                        
                        if cur_delta_method == self.DeltaMethod.R1 and not use_hue:
                            if t == 0:
                                last_pixel_data = np.zeros_like(pixel_data)
                            else:
                                pixel_data = np.copy(pixel_data - last_pixel_data)

                        res = self._jp2_encoding(pixel_data, config["folder"], os.path.join(config["image_folder"], f"{name}.jp2"), cur_ratio, use_hue, config["color_method"], config["color_type"], self.DeltaMethod.NONE if t == 0 else cur_delta_method, last_pixel_data)
                        if use_hue and cur_delta_method == self.DeltaMethod.N1:
                            last_pixel_data = np.copy(pixel_data)

                        if cur_delta_method == self.DeltaMethod.R1:
                            tmp = self._jp2_decoding(res, config["folder"], use_hue, config["color_method"], config["color_type"], self.DeltaMethod.NONE if t == 0 else cur_delta_method, last_pixel_data)
                            if use_hue:
                                last_pixel_data = tmp
                            else:
                                last_pixel_data = last_pixel_data + tmp
                    
                    
                    case self.CompressionMethod.NUMPY:
                        if cur_delta_method == self.DeltaMethod.R1:
                            if t == 0:
                                last_pixel_data = np.zeros_like(pixel_data)
                            else:
                                pixel_data = np.copy(pixel_data - last_pixel_data)
                    
                        np.save(os.path.join(config["folder"], config["image_folder"],name), pixel_data)
                        res["jp2_path"] = os.path.join(config["image_folder"],f"{name}.npy")
                    
                        if cur_delta_method == self.DeltaMethod.R1:
                            tmp = np.load(os.path.join(config["folder"], res["jp2_path"]))
                            last_pixel_data = last_pixel_data + tmp
                    
                    
                    case self.CompressionMethod.NUMPY16:
                        if cur_delta_method == self.DeltaMethod.R1:
                            if t == 0:
                                last_pixel_data = np.zeros_like(pixel_data)
                            else:
                                pixel_data = np.copy(pixel_data - last_pixel_data)
                        type = np.uint16
                        level_max = pixel_data.max()
                        level_min = pixel_data.min()
                        maxval = np.iinfo(type).max
                        pixel_data = (pixel_data - level_min) / (level_max - level_min)
                        pixel_data = (pixel_data * maxval).astype(type)
                        np.save(os.path.join(config["folder"], config["image_folder"],name), pixel_data)
                        res["jp2_path"] = os.path.join(config["image_folder"],f"{name}.npy")
                        res["level_min"] = level_min
                        res["level_max"] = level_max

                        if cur_delta_method == self.DeltaMethod.R1:
                            tmp = np.load(os.path.join(config["folder"], res["jp2_path"]))
                            type = np.uint16
                            level_max = res["level_max"]
                            level_min = res["level_min"]
                            maxval = np.iinfo(type).max
                            tmp = tmp.astype(np.float32) / maxval
                            tmp = tmp * (level_max - level_min) + level_min
                            last_pixel_data = last_pixel_data + tmp


                    case self.CompressionMethod.NUMPY8:
                        if cur_delta_method == self.DeltaMethod.R1:
                            if t == 0:
                                last_pixel_data = np.zeros_like(pixel_data)
                            else:
                                pixel_data = np.copy(pixel_data - last_pixel_data)
                        type = np.uint8
                        level_max = pixel_data.max()
                        level_min = pixel_data.min()
                        maxval = np.iinfo(type).max
                        pixel_data = (pixel_data - level_min) / (level_max - level_min)
                        pixel_data = (pixel_data * maxval).astype(type)
                        np.save(os.path.join(config["folder"], config["image_folder"],name), pixel_data)
                        res["jp2_path"] = os.path.join(config["image_folder"],f"{name}.npy")
                        res["level_min"] = level_min
                        res["level_max"] = level_max

                        if cur_delta_method == self.DeltaMethod.R1:
                            tmp = np.load(os.path.join(config["folder"], res["jp2_path"]))
                            type = np.uint8
                            level_max = res["level_max"]
                            level_min = res["level_min"]
                            maxval = np.iinfo(type).max
                            tmp = tmp.astype(np.float32) / maxval
                            tmp = tmp * (level_max - level_min) + level_min
                            last_pixel_data = last_pixel_data + tmp


                    case self.CompressionMethod.SZ3:
                        if cur_delta_method == self.DeltaMethod.R1:
                            if t == 0:
                                last_pixel_data = np.zeros_like(pixel_data)
                            else:
                                pixel_data = np.copy(pixel_data - last_pixel_data)
                        sz3_ratio = 0.00013*cur_ratio
                        res = self._sz3_encoding(pixel_data, config["folder"], os.path.join(config["image_folder"], name), sz3_ratio)

                        if cur_delta_method == self.DeltaMethod.R1:
                            tmp = self._sz3_decoding(res, config["folder"])
                            last_pixel_data = last_pixel_data + tmp

                        
                    case _:
                        print("ERROR: Unknown compression method")
                        exit(-1)
                
                res["color_method"] = config["color_method"]
                res["color_type"] = config["color_type"]
                res["data_variable"] = d
                res["time"] = t
                res["pressure"] = pressure
                res["method"] = cur_method
                res["delta_method"] = cur_delta_method
                meta.jpeg_metas.append(res)
        print(f"done with level {pressure}")
        return meta
    
    def _lon_lat_to_cartesian(self, lon, lat, R = 1.):
        """
        calculates lon, lat coordinates of a point on a sphere with
        radius R
        """
        lon_r = np.radians(lon)
        lat_r = np.radians(lat)

        x = R * np.cos(lat_r) * np.cos(lon_r)
        y = R * np.cos(lat_r) * np.sin(lon_r)
        z = R * np.sin(lat_r)
        return x,y,z

    def _mapc2p(self,xc,yc):
        n = xc.shape[0]
        r1 = 1
        d = np.maximum(np.abs(xc),np.abs(yc))
        d = np.maximum(d, np.full(n,1e-10))
        #D = r1 * d/np.sqrt(2)
        D = r1 * d*(2 - d)/np.sqrt(2)
        #R = r1 * d
        R = r1*np.full(n,1.)
        center = D - np.sqrt(np.power(R,2) - np.power(D,2))
        xp = D/d * np.abs(xc)
        yp = D/d * np.abs(yc)
        ij = np.where(np.abs(yc)>=np.abs(xc))[0]
        yp[ij] = center[ij] + np.sqrt(np.power(R[ij],2) - np.power(xp[ij],2))
        ij = np.where(np.abs(xc)>=np.abs(yc))[0]
        xp[ij] = center[ij] + np.sqrt(np.power(R[ij], 2) - np.power(yp[ij],2))
        xp = np.sign(xc) * xp
        yp = np.sign(yc) * yp
        return (xp,yp)

    def _mapsphere(self,xc,yc):
        xc = (xc/180)*2 -1
        yc = yc/90
        lower_idxs = np.where(xc < -1)[0]
        xc[lower_idxs] = -2 - xc[lower_idxs]
        xp,yp = self._mapc2p(xc,yc)
        zp = np.sqrt(1 - (np.power(xp,2) + np.power(yp,2)))
        zp[lower_idxs] *= -1.
        return xp,yp,zp

    def _create_mapping_file(self, filename, x_pixels, y_pixels, lons, lats):
        mapping = np.zeros((y_pixels, x_pixels),dtype=np.int32)
        xs, ys, zs = self._lon_lat_to_cartesian(lons, lats)
        tree = cKDTree(np.column_stack((xs, ys, zs)))
        x_pos = np.linspace(-180,180,x_pixels, False)
        y_pos = np.linspace(-90,90, y_pixels, True)
        xx, yy = np.meshgrid(np.arange(0,x_pixels,1, dtype=np.int32), np.arange(0,y_pixels,1,dtype=np.int32))
        xx = xx.flatten()
        yy = yy.flatten()
        xp,yp,zp = self._mapsphere(x_pos[xx], y_pos[yy])
        _, idxs = tree.query(np.column_stack((xp,yp,zp)), k=1)
        mapping[yy,xx] = idxs
        
        x_pos = np.linspace(-180,180,x_pixels, False)
        y_pos = np.linspace(-90,90, y_pixels, True)
        xx, yy = np.meshgrid(np.arange(0,x_pixels,1, dtype=np.int32), np.arange(0,y_pixels,1,dtype=np.int32))
        xx = xx.flatten()
        yy = yy.flatten()
        
        xs, ys, zs = self._lon_lat_to_cartesian(lons, lats)
        xp,yp,zp = self._mapsphere(x_pos[xx], y_pos[yy])
        tree = cKDTree(np.column_stack((xp, yp, zp)))
        
        neight_count = 200
        
        try:
            d, idxs = tree.query(np.column_stack((xs,ys,zs)), k=neight_count)
            #print(f"min dist: {d.flatten().min()}")
            #print(f"max dist: {d.flatten().max()}")
            #sparse_mat_adj = scipy.sparse.coo_matrix((np.full(len(idxs.flatten()),1), (np.repeat(np.arange(0,len(lons), 1, dtype=np.int32), neight_count), idxs.flatten())))
            #perm = scipy.sparse.csgraph.maximum_bipartite_matching(sparse_mat.tocsr(), perm_type="column")
            sparse_mat = scipy.sparse.coo_matrix((d.flatten(), (np.repeat(np.arange(0,len(lons), 1, dtype=np.int32), neight_count), idxs.flatten())))
            row_ind, col_ind = scipy.sparse.csgraph.min_weight_full_bipartite_matching(sparse_mat)
        except ValueError:
            print("ERROR: Could not calculate min weight full bipartite matching, this is most likely due to the fact tha the input data is not distribued nicely.")
        

        inverse_mapping = np.empty((len(lons),2), np.int32)
        #inverse_mapping[:,:] = np.dstack((yy[perm],xx[perm]))
        inverse_mapping[row_ind,:] = np.dstack((yy[col_ind],xx[col_ind]))

        mapping[inverse_mapping[:,0],inverse_mapping[:,1]] = np.arange(0,len(lons), 1, dtype=np.int32)

        mask = np.zeros((y_pixels, x_pixels), dtype=np.int16)
        mask[inverse_mapping[:,0],inverse_mapping[:,1]] = 1


        self._store_mapping(filename, inverse_mapping)
        return mapping, mask
    
    def _create_mapping_file_lp(self, filename, x_pixels, y_pixels, lons, lats):
        n = len(lons)
        k = 50
        d = n*k
        
        x_pos = np.linspace(-180,180,x_pixels, False)
        y_pos = np.linspace(-90,90, y_pixels, True)
        xx, yy = np.meshgrid(np.arange(0,x_pixels,1, dtype=np.int32), np.arange(0,y_pixels,1,dtype=np.int32))
        xx = xx.flatten()
        yy = yy.flatten()
        xs, ys, zs = self._lon_lat_to_cartesian(lons, lats)
        xp,yp,zp = self._mapsphere(x_pos[xx], y_pos[yy])
        tree = cKDTree(np.column_stack((xp, yp, zp)))
        dist, idxs = tree.query(np.column_stack((xs,ys,zs)), k=k)

        m = gp.Model("mapping")
        x = m.addMVar(shape=d, vtype=GRB.CONTINUOUS, lb=0.,ub=1., name="x")
        row = np.empty(0, dtype=np.int32)
        col = np.empty(0, dtype=np.int32)
        row, col = ctt.find_indxs(idxs,x_pixels*y_pixels,k)
        val = np.full_like(row,1)
        A1 = scipy.sparse.csr_matrix((val, (row,col)), shape=(x_pixels*y_pixels, d))
        A2 = scipy.sparse.csr_matrix((np.full(d,1), (np.repeat(np.arange(0,n,1,dtype=np.int32),k), np.arange(0,d,1,dtype=np.int32))), shape=(n,d))
        m.addConstr(A1 @ x == np.full(x_pixels*y_pixels, 1))
        m.addConstr(A2 @ x >= np.full(n, 1))
        m.setObjective(dist.flatten() @ x, GRB.MINIMIZE)
        m.optimize()
        mapping = np.zeros((y_pixels, x_pixels),dtype=np.int32)
        mapping = mapping.flatten()
        mapping[idxs.flatten()[np.where(x.X == 1)[0]]] = np.where(x.X == 1)[0] // k
        mapping.shape = (y_pixels, x_pixels)
        
        i = np.arange(0,mapping.shape[0], 1, dtype=np.int32)
        j = np.arange(0,mapping.shape[1], 1, dtype=np.int32)
        jj, ii = np.meshgrid(j,i)
        inverse_mapping = np.empty((len(lons),2), np.int32)
        inverse_mapping[mapping[ii,jj],:] = np.dstack((ii,jj))

        self._store_mapping(filename, inverse_mapping)
        return mapping, mapping


    def _create_inverse_mapping_file(self, filename):
        mapping = self._load_mapping(filename)
        return mapping
    

    def _compress_idxs(self, idxs):
        m = idxs.max()
        shape = idxs.shape
        
        # normalize the lats, to -int16.max and int16.max
        entropy_model = constriction.stream.model.Uniform(m+1)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(idxs.flatten(), entropy_model)

        compressed = encoder.get_compressed()
        return {"m" : m, "shape" : shape, "idxs" : compressed}

    def _decopmress_idxs(self, d):
        shape = d["shape"]
        n = 1
        for s in shape:
            n *= s
        m = d["m"]
        compress = d["idxs"]
        entropy_model = constriction.stream.model.Uniform(m+1)
        decoder = constriction.stream.stack.AnsCoder(compress)
        idxs = decoder.decode(entropy_model, n).astype(np.int32)
        idxs.shape = shape
        return idxs



    def _store_mapping(self, filename, data):
        data_file = open(filename, "wb")
        m = data.max()
        data_file.write(np.int32(data.shape[0]))
        data_file.write(np.int32(data.shape[1]))
        data_file.write(np.int32(m))
        # normalize the lats, to -int16.max and int16.max
        entropy_model = constriction.stream.model.Uniform(m+1)
        encoder = constriction.stream.stack.AnsCoder()
        encoder.encode_reverse(data.flatten(), entropy_model)

        compressed = encoder.get_compressed()
        data_file.write(np.uint32(4*len(compressed)))
        data_file.write(compressed)
        data_file.close()

    def _load_mapping(self, filename):
        data_file = open(filename, "rb")
        nx = np.frombuffer(data_file.read(4), dtype=np.int32)[0]
        ny = np.frombuffer(data_file.read(4), dtype=np.int32)[0]
        m = np.frombuffer(data_file.read(4), dtype=np.int32)[0]
        l = np.frombuffer(data_file.read(4), dtype=np.uint32)[0]
        compress = np.frombuffer(data_file.read(l), np.uint32)
        entropy_model = constriction.stream.model.Uniform(m+1)
        decoder = constriction.stream.stack.AnsCoder(compress)
        data = decoder.decode(entropy_model, nx*ny).astype(np.int32)
        data.shape = (nx,ny)
        return data

    

    def _jp2_encoding(self, input_arr: np.ndarray, work_dir: str, output_path: str, compression_ratio: int, use_hue: bool, color_method, color_type, delta_method, last_arr):
        """
        input_arr: input array, should be 2D with shape (nlat, nlon)
        output_path: path of resulting jpeg2000 file, example "data/var1_t0_level850.jp2"
        compression_ratio: target compression ratio
        """
        #use_hue = True
        assert input_arr.dtype == np.float32
        if not use_hue or color_method != self.ColorMethod.XYZ:
            nlat, nlon = input_arr.shape
        else:
            dummy, nlat, nlon = input_arr.shape
        channels = 1
        if use_hue:
            maxval = np.iinfo(color_type).max
            if color_method == self.ColorMethod.HSV or color_method == self.ColorMethod.XYZ:
                channels = 3
            else:
                channels = 2
        # number of 2D fields we want to compress in one JP2 file
        ntileh, ntilew = 1, 1

        jp2 = glymur.Jp2k(os.path.join(work_dir, output_path), irreversible=True,
                            shape=(nlat*ntileh, nlon*ntilew, channels), tilesize=(nlat, nlon),
                            # compression ratio is calculated based on uint16, but we assume the input data has the type float32 
                            cratios=[compression_ratio // 2])

        data_ref = input_arr
        # convert float32 to uint16
        if not use_hue or color_method != self.ColorMethod.XYZ:
            level_max = input_arr.max()
            level_min = input_arr.min()
            if use_hue:
                maxval = np.iinfo(color_type).max
            else:
                maxval = np.iinfo(np.uint16).max
            if level_max == level_min:
                data_ref = data_ref*0.
            else:
                data_ref = (data_ref - level_min) / (level_max - level_min)

        if use_hue and delta_method != self.DeltaMethod.NONE and not color_method == self.ColorMethod.XYZ:
            last_data_max = last_arr.max()
            last_data_min = last_arr.min()
            last_data_ref = (last_arr - last_data_min) / (last_data_max - last_data_min)
        if use_hue:
            data_loc = np.zeros((nlat,nlon,channels))
            match color_method:
                case self.ColorMethod.CIRCLE:
                    r,g = self._circle_to_rgb(data_ref.flatten())
                    r.shape = data_ref.shape
                    g.shape = data_ref.shape
                    data_loc[:,:,0] = r
                    data_loc[:,:,1] = g
                    if delta_method != self.DeltaMethod.NONE:                        
                        last_data_loc = np.zeros((nlat,nlon,channels))
                        last_r,last_g = self._circle_to_rgb(last_data_ref.flatten())
                        last_r.shape = last_data_ref.shape
                        last_g.shape = last_data_ref.shape
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g

                        delta_data_loc = data_loc - last_data_loc
                        delta_min = delta_data_loc.min()
                        delta_max = delta_data_loc.max()
                        delta_data_loc = (delta_data_loc - delta_min) / (delta_max - delta_min)
                        data_loc = delta_data_loc
                case self.ColorMethod.LUIGI:
                    res = self._snake_to_rgb(data_ref.flatten())
                    r = res[:,0]
                    g = res[:,1]
                    r.shape = data_ref.shape
                    g.shape = data_ref.shape
                    data_loc[:,:,0] = r
                    data_loc[:,:,1] = g
                    if delta_method != self.DeltaMethod.NONE:
                        last_data_loc = np.zeros((nlat,nlon,channels))
                        last_res = self._snake_to_rgb(last_data_ref.flatten())
                        last_r = last_res[:,0]
                        last_g = last_res[:,1]
                        last_r.shape = last_data_ref.shape
                        last_g.shape = last_data_ref.shape
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g

                        delta_data_loc = data_loc - last_data_loc
                        delta_min = delta_data_loc.min()
                        delta_max = delta_data_loc.max()
                        delta_data_loc = (delta_data_loc - delta_min) / (delta_max - delta_min)
                        data_loc = delta_data_loc
                case self.ColorMethod.HSV:
                    r,g,b = self._hsv_to_rgb(data_ref, 1, 1)
                    data_loc[:,:,0] = r
                    data_loc[:,:,1] = g
                    data_loc[:,:,2] = b
                    if delta_method != self.DeltaMethod.NONE:
                        last_data_loc = np.zeros((nlat,nlon,channels))
                        last_r,last_g,last_b = self._hsv_to_rgb(last_data_ref,1,1)
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g
                        last_data_loc[:,:,2] = last_b

                        delta_data_loc = data_loc - last_data_loc
                        delta_min = delta_data_loc.min()
                        delta_max = delta_data_loc.max()
                        delta_data_loc = (delta_data_loc - delta_min) / (delta_max - delta_min)
                        data_loc = delta_data_loc
                case self.ColorMethod.XYZ:
                    lon = data_ref[0]
                    lat = data_ref[1]
                    lon_min = lon.min()
                    lat_min = lat.min()
                    lon_max = lon.max()
                    lat_max = lat.max()
                    lon = (lon - lon_min) / (lon_max - lon_min)
                    lat = (lat - lat_min) / (lat_max - lat_min)
                    r,g,b = self._lon_lat_to_rgb(lon,lat)
                    data_loc[:,:,0] = r
                    data_loc[:,:,1] = g
                    data_loc[:,:,2] = b
                    level_max = 0 # dummy variable
                    level_min = 0 # dummy variable
                    if delta_method != self.DeltaMethod.NONE:
                        last_data_lon = last_arr[0]
                        last_data_lat = last_arr[1]
                        last_data_lon_min = last_data_lon.min()
                        last_data_lon_max = last_data_lon.max()
                        last_data_lat_min = last_data_lat.min()
                        last_data_lat_max = last_data_lat.max()
                        last_data_lon = (last_data_lon - last_data_lon_min) / (last_data_lon_max - last_data_lon_min)
                        last_data_lat = (last_data_lat - last_data_lat_min) / (last_data_lat_max - last_data_lat_min)
                        last_r,last_g,last_b = self._lon_lat_to_rgb(last_data_lon, last_data_lat)
                        last_data_loc = np.zeros((nlat,nlon,channels))
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g
                        last_data_loc[:,:,2] = last_b

                        delta_data_loc = data_loc - last_data_loc
                        delta_min = delta_data_loc.min()
                        delta_max = delta_data_loc.max()
                        delta_data_loc = (delta_data_loc - delta_min) / (delta_max - delta_min)
                        data_loc = delta_data_loc
                case _:
                    print(f"Invalid color method encountered in jp2 encoding")
                    exit(-1)
                
            data_ref = data_loc
            data_ref = (data_ref * maxval).astype(color_type)
        else:
            data_ref = (data_ref * maxval).astype(np.uint16)
        jp2[:] = data_ref

        array_size = input_arr.size * input_arr.itemsize
        file_size = os.path.getsize(os.path.join(work_dir,output_path))
        real_cratio = array_size / file_size
        #print(f"File size: {file_size / (1024 * 1024)} MB, jp2_encoding cratio: {real_cratio}")
        res = {"jp2_path": output_path, "level_max": level_max, "level_min": level_min,
                "jp2_size": file_size, "target_cratio": compression_ratio, "real_cratio": real_cratio}
        if use_hue and color_method == self.ColorMethod.XYZ:
            res["lon_min"] = lon_min
            res["lon_max"] = lon_max
            res["lat_min"] = lat_min
            res["lat_max"] = lat_max
        if use_hue and delta_method != self.DeltaMethod.NONE:
            res["delta_min"] = delta_min
            res["delta_max"] = delta_max
        return res

    def _jp2_decoding(self, res, work_dir, use_hue: bool, color_method, color_type, delta_method, last_arr):
        #use_hue = True
        maxval = np.iinfo(np.uint16).max
        if use_hue:
            maxval = np.iinfo(color_type).max
        
        level_max = res["level_max"]
        level_min = res["level_min"]
        jp2 = glymur.Jp2k(os.path.join(work_dir,res["jp2_path"]))
        data_decomp = jp2[:].astype(np.float32) / maxval
        if use_hue:
            match color_method:
                case self.ColorMethod.CIRCLE:
                    if delta_method != self.DeltaMethod.NONE:
                        channels = 2
                        nlat, nlon = last_arr.shape
                        last_data_max = last_arr.max()
                        last_data_min = last_arr.min()
                        last_data_ref = (last_arr - last_data_min) / (last_data_max - last_data_min)
                        last_data_loc = np.zeros((nlat,nlon,channels))

                        last_data_loc = np.zeros((nlat,nlon,channels))
                        last_r,last_g = self._circle_to_rgb(last_data_ref.flatten())
                        last_r.shape = last_data_ref.shape
                        last_g.shape = last_data_ref.shape
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g

                        data_decomp = data_decomp * (res["delta_max"] - res["delta_min"]) + res["delta_min"]
                        data_decomp = last_data_loc + data_decomp

                    data_loc = self._rgb_to_circle(data_decomp[:,:,0].flatten(), data_decomp[:,:,1].flatten())
                    data_loc.shape = (data_decomp.shape[0], data_decomp.shape[1])

                case self.ColorMethod.LUIGI:
                    if delta_method != self.DeltaMethod.NONE:
                        channels = 2
                        nlat, nlon = last_arr.shape
                        last_data_max = last_arr.max()
                        last_data_min = last_arr.min()
                        last_data_ref = (last_arr - last_data_min) / (last_data_max - last_data_min)
                        last_data_loc = np.zeros((nlat,nlon,channels))

                        last_res = self._snake_to_rgb(last_data_ref.flatten())
                        last_r = last_res[:,0]
                        last_g = last_res[:,1]
                        last_r.shape = last_data_ref.shape
                        last_g.shape = last_data_ref.shape
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g

                        data_decomp = data_decomp * (res["delta_max"] - res["delta_min"]) + res["delta_min"]
                        data_decomp = last_data_loc + data_decomp

                    data_loc = self._rgb_to_snake(data_decomp[:,:,0].flatten(), data_decomp[:,:,1].flatten())
                    data_loc.shape = (data_decomp.shape[0], data_decomp.shape[1])

                case self.ColorMethod.HSV:
                    if delta_method != self.DeltaMethod.NONE:
                        channels = 3
                        nlat, nlon = last_arr.shape
                        last_data_max = last_arr.max()
                        last_data_min = last_arr.min()
                        last_data_ref = (last_arr - last_data_min) / (last_data_max - last_data_min)
                        last_data_loc = np.zeros((nlat,nlon,channels))

                        last_r,last_g,last_b = self._hsv_to_rgb(last_data_ref,1,1)
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g
                        last_data_loc[:,:,2] = last_b

                        data_decomp = data_decomp * (res["delta_max"] - res["delta_min"]) + res["delta_min"]
                        data_decomp = last_data_loc + data_decomp

                    data_loc = np.empty((data_decomp.shape[0], data_decomp.shape[1]))
                    data_loc[:,:] = self._rgb_to_hue(data_decomp[:,:,0], data_decomp[:,:,1], data_decomp[:,:,2])
                case self.ColorMethod.XYZ:
                    if delta_method != self.DeltaMethod.NONE:
                        channels = 3
                        dummy, nlat, nlon = last_arr.shape
                        last_data_lon = last_arr[0]
                        last_data_lat = last_arr[1]
                        last_data_lon_max = last_data_lon.max()
                        last_data_lon_min = last_data_lon.min()
                        last_data_lat_max = last_data_lat.max()
                        last_data_lat_min = last_data_lat.min()
                        last_data_lon = (last_data_lon - last_data_lon_min) / (last_data_lon_max - last_data_lon_min)
                        last_data_lat = (last_data_lat - last_data_lat_min) / (last_data_lat_max - last_data_lat_min)
                        last_data_loc = np.zeros((nlat,nlon,channels))

                        last_r,last_g,last_b = self._lon_lat_to_rgb(last_data_lon, last_data_lat)
                        last_data_loc[:,:,0] = last_r
                        last_data_loc[:,:,1] = last_g
                        last_data_loc[:,:,2] = last_b

                        data_decomp = data_decomp * (res["delta_max"] - res["delta_min"]) + res["delta_min"]
                        data_decomp = last_data_loc + data_decomp
                    lon = np.empty((data_decomp.shape[0], data_decomp.shape[1]))
                    lat = np.empty((data_decomp.shape[0], data_decomp.shape[1]))
                    lon,lat = self._rgb_to_lon_lat(data_decomp[:,:,0], data_decomp[:,:,1], data_decomp[:,:,2])
                    lon = lon * (res["lon_max"] - res["lon_min"]) + res["lon_min"]
                    lat = lat * (res["lat_max"] - res["lat_min"]) + res["lat_min"]
                    data_loc = np.stack((lon,lat))
                case _:
                    print(f"Invalid color method encountered in jp2 decoding.")
                    exit(-1)

            data_decomp = data_loc
        if not use_hue or color_method != self.ColorMethod.XYZ:
            data_decomp = data_decomp * (level_max - level_min) + level_min
        return data_decomp
    
    def _sz3_encoding(self, input_arr: np.ndarray, work_dir: str, output_path: str, max_rel_err: float):
        sz = SZ(f"{os.path.dirname(os.path.realpath(__file__))}/SZ3/install/lib/libSZ3c.so")
        data_cmpr, cmpr_ratio = sz.compress(input_arr, 1, 0, max_rel_err, 0)
        np.save(os.path.join(work_dir,output_path), data_cmpr)
        res = {"real_cratio": cmpr_ratio, "path" : output_path + ".npy", "shape" : input_arr.shape}
        return res
    
    def _sz3_decoding(self, res, work_dir):
        sz = SZ(f"{os.path.dirname(os.path.realpath(__file__))}/SZ3/install/lib/libSZ3c.so")
        data_cmpr = np.load(os.path.join(work_dir, res["path"]))
        data_dec = sz.decompress(data_cmpr, res["shape"], np.float32)
        return data_dec

    def _hsv_to_rgb(self, h,s,v):
        h = h*360
        def f(n):
            k = np.mod(n+h/60, 6)
            return v - v*s*np.maximum(0, np.minimum(np.minimum(k,4-k),1))
        return (f(5), f(3), f(1))
    
    def _rgb_to_hue(self, r,g,b):
        s = r.shape
        r = r.flatten()
        g = g.flatten()
        b = b.flatten()
        rgb = np.column_stack((r,g,b))
        x_max_ind = np.argmax(rgb, axis=1)
        x_min_ind = np.argmin(rgb, axis=1)
        c = rgb[np.arange(0,rgb.shape[0],1, dtype=np.int32),x_max_ind] - rgb[np.arange(0,rgb.shape[0],1, dtype=np.int32),x_min_ind]
        u = 60*np.mod((g-b)/c,6)
        v = 60*((b-r)/c+2)
        w = 60*((r-g)/c+4)
        uvw = np.column_stack((u,v,w))
        h = uvw[np.arange(0,uvw.shape[0],1, dtype=np.int32),x_max_ind]
        h[np.where(c < 1e-3)[0]] = 0.
        h.shape = s
        h = h/360.
        return h
    
    def _snake_to_rgb(self, x):
        idx = np.floor((self.snake_data.shape[0]-1)*x).astype(np.int32)
        next_idx = np.mod(idx+1, self.snake_data.shape[0])
        a = idx.astype(np.float32)/(self.snake_data.shape[0]-1)
        b = next_idx.astype(np.float32)/(self.snake_data.shape[0]-1)
        delta = np.abs((x - a) / (b - a))
        return (self.snake_data[idx,:]  + (self.snake_data[np.mod(idx+1, self.snake_data.shape[0]),:] - self.snake_data[idx,:])*delta[:,np.newaxis])

    def _rgb_to_snake(self, r,g):
        res = np.full_like(r,-1)

        tree = cKDTree(np.column_stack((self.snake_data[:,0], self.snake_data[:,1])))
        _, idxs = tree.query(np.column_stack((r,g)), k=1)
        prev_idxs = np.mod(idxs -1, self.snake_data.shape[0])
        next_idxs = np.mod(idxs +1, self.snake_data.shape[0])
        v = np.vstack((r-self.snake_data[idxs,0], g-self.snake_data[idxs,1])).T
        def proj(a,b):
            return (a*b).sum(1)
        proj_next = proj(self.snake_data[next_idxs,0:2] - self.snake_data[idxs,0:2], v)/proj(self.snake_data[next_idxs,0:2] - self.snake_data[idxs,0:2], self.snake_data[next_idxs,0:2] - self.snake_data[idxs,0:2])
        proj_prev = proj(self.snake_data[prev_idxs,0:2] - self.snake_data[idxs,0:2], v)/proj(self.snake_data[prev_idxs,0:2] - self.snake_data[idxs,0:2], self.snake_data[prev_idxs,0:2] - self.snake_data[idxs,0:2])
        path_length = self.snake_data.shape[0]

        proj_both_neg = ((proj_prev < -1e-5) & (proj_next < -1e-5)).nonzero()[0]
        proj_next_neg = ((proj_next < -1e-5) & (proj_prev >= -1e-5)).nonzero()[0]
        proj_prev_neg = ((proj_prev < -1e-5) & (proj_next >= -1e-5)).nonzero()[0]
        proj_none_neg = ((proj_prev >= -1e-5) & (proj_next >= -1e-5)).nonzero()[0]

        res[proj_both_neg] = idxs[proj_both_neg]/(self.snake_data.shape[0]-1)

        res[proj_next_neg] = idxs[proj_next_neg]/(self.snake_data.shape[0]-1) - proj_prev[proj_next_neg]/path_length

        res[proj_prev_neg] = idxs[proj_prev_neg]/(self.snake_data.shape[0]-1) + proj_next[proj_prev_neg]/path_length
        
        proj_none_neg_a = proj_none_neg[(proj_next[proj_none_neg] > proj_prev[proj_none_neg]).nonzero()[0]]
        proj_none_neg_b = proj_none_neg[(proj_next[proj_none_neg] <= proj_prev[proj_none_neg]).nonzero()[0]]
        res[proj_none_neg_a] = idxs[proj_none_neg_a]/(self.snake_data.shape[0]-1) + proj_next[proj_none_neg_a]/path_length
        res[proj_none_neg_b] = idxs[proj_none_neg_b]/(self.snake_data.shape[0]-1) - proj_prev[proj_none_neg_b]/path_length
        
        return res


    def _circle_to_rgb(self, x):
        r = np.cos(x*2*np.pi)
        g = np.sin(x*2*np.pi)
        r = (r+1.)/2
        g = (g+1.)/2
        return (r,g)


    def _rgb_to_circle(self, r,g):
        r = r*2 -1
        g = g*2 -1
        return np.mod(np.arctan2(g,r)/(2*np.pi)+1,1)
    
    def _lon_lat_to_rgb(self, lon,lat):
        lon_rad = (lon-0.5)*2*np.pi
        lat_rad = (lat-0.5)*np.pi
        

        r = np.cos(lat_rad) * np.cos(lon_rad)
        g = np.cos(lat_rad) * np.sin(lon_rad)
        b = np.sin(lat_rad)
        r = (r+1)/2
        g = (g+1.)/2.
        b = (b+1)/2
        return r,g,b

    def _rgb_to_lon_lat(self, r,g,b):
        r = 2*r - 1
        g = 2.*g - 1.
        b = 2*b - 1

        hyp = np.sqrt(r**2 + g**2)
        lat_rad = np.arctan2(b, hyp)
        lon_rad = np.arctan2(g, r)

        lat = (lat_rad + np.pi/2)/(np.pi)
        lon = (lon_rad + np.pi)/(2*np.pi)
        
        return lon, lat
        
