import numpy as np
from tabulate import tabulate


def calc_l1(dif):
    return np.abs(dif).sum()/dif.size

def calc_rmse(dif):
    dif = np.power(dif, 2)
    return np.sqrt(dif.sum()/dif.size)

def calc_linf(dif):
    return np.abs(dif).max()

def calc_dif(a_dict, b_dict, xyz_mapping):
    dif_dict = dict()
    for d in a_dict:
        if (d == "lat" or d == "lon") and xyz_mapping:
            continue
        a = a_dict[d]
        b = b_dict[d]
        if d == "lon":
            v1 = a-b
            v2 = np.where(v1 < 0, -np.mod(360 + v1, 360), np.mod(360. - v1,360))
            dif = v2
            k = np.where(np.abs(v1) < np.abs(v2))
            dif[k] = v1[k]
        else:
            dif = a-b
        dif_dict[d] = dif
    
    if "lon" in a_dict and "lat" in a_dict:
        lon_rad_a = np.radians(a_dict["lon"].astype(np.float64))
        lat_rad_a = np.radians(a_dict["lat"].astype(np.float64))
        lon_rad_b = np.radians(b_dict["lon"].astype(np.float64))
        lat_rad_b = np.radians(b_dict["lat"].astype(np.float64))
        dlon = np.abs(lon_rad_a - lon_rad_b)
        dlat = np.abs(lat_rad_a - lat_rad_b)
        dif = 2*np.arcsin(np.sqrt(np.sin(dlat/2)**2 + (1-np.sin(dlat/2)**2 - np.sin((lat_rad_a + lat_rad_b)/2)**2)*np.sin(dlon/2)**2))
        #a = np.sin(dlat/2.0)**2 + np.cos(lat_rad_a) * np.cos(lat_rad_b) * np.sin(dlon/2.0)**2
        #dif = 2 * np.arcsin(np.sqrt(a))
        dif = np.degrees(dif)
        dif_dict["lon-lat"] = dif
    return dif_dict

def pretty_print(a_dict,b_dict, xyz_mapping):
    print("")
    print("")
    print("")
    print("--------------------------------------------------------------------------------")
    print("Compression-errors per data variable for all levels and all timesteps.")
    print("--------------------------------------------------------------------------------")
    print("")
    data = []
    data_dict = dict()
    dif_dict = calc_dif(a_dict, b_dict,xyz_mapping)
    for d in dif_dict:
        dif = dif_dict[d]
        l1 = calc_l1(dif)
        rmse = calc_rmse(dif)
        linf = calc_linf(dif) 
        data.append([d, l1, rmse, linf])
        data_dict[d] = {"l1" : l1, "rmse" : rmse, "linf" : linf}
    print(tabulate(data, headers=["data variable", "l1", "rmse", "linf"]))
    return data_dict


def er_over_time(a_dict, b_dict, xyz_mapping):
    print("")
    print("")
    print("")
    print("--------------------------------------------------------------------------------")
    print("Compression-errors per data variable for all levels given for each timestep.")
    print("--------------------------------------------------------------------------------")
    print("")

    print("l1 error over time")
    print("------------------")
    data = []
    headers = ["data variable"] + [f"step {x}" for x in list(range(0,a_dict[next(iter(a_dict))].shape[0]))]
    dif_dict = calc_dif(a_dict,b_dict, xyz_mapping)
    for d in dif_dict:
        ers = [d]
        for t in range(dif_dict[d].shape[0]):
            dif = dif_dict[d][t]
            ers.append(calc_l1(dif))
        data.append(ers)
    print(tabulate(data, headers=headers))
    print("")


    print("rmse error over time")
    print("--------------------")
    data = []
    headers = ["data variable"] + [f"step {x}" for x in list(range(0,a_dict[next(iter(a_dict))].shape[0]))]
    dif_dict = calc_dif(a_dict,b_dict, xyz_mapping)
    for d in dif_dict:
        ers = [d]
        for t in range(dif_dict[d].shape[0]):
            dif = dif_dict[d][t]
            ers.append(calc_rmse(dif))
        data.append(ers)
    print(tabulate(data, headers=headers))
    print("")


    print("linf error over time")
    print("--------------------")
    data = []
    headers = ["data variable"] + [f"step {x}" for x in list(range(0,a_dict[next(iter(a_dict))].shape[0]))]
    dif_dict = calc_dif(a_dict,b_dict, xyz_mapping)
    for d in dif_dict:
        ers = [d]
        for t in range(dif_dict[d].shape[0]):
            dif = dif_dict[d][t]
            ers.append(calc_linf(dif))
        data.append(ers)
    print(tabulate(data, headers=headers))
    print("")
