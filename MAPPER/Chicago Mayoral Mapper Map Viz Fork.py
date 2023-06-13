#import necessary packages
import csv
import kmapper as km
import numpy as np
import matplotlib.pyplot as plt
import sys, os
from kmapper.plotlyviz import plotlyviz
import networkx as nx
from sklearn import preprocessing, datasets
from numpy import genfromtxt
import pandas as pd
import geopandas as gpd
from statistics import mode
import folium
import mapclassify
import IPython
import re

#first read in chicago data
chicago_mayoral = pd.read_csv("/Users/emariedelanuez/Chicago-Mayoral-Mapper-Map-Viz-Fork/keppler_mapper_data/Chicago_mayoral.csv")
for col in chicago_mayoral.columns: print(col)
#print(chicago_mayoral.head())

# separate VAP columns with Salary columns
vap_cols = [col for col in chicago_mayoral if "VAP_pct" in col]
salary_cols = [col for col in chicago_mayoral.columns if "K_pct" in col or col == "200K_MORE_pct"]

#create two dataframes that run through the entire mapper process
#maps demographic shares in each precinct to the percent of support for Chuy Garcia.
chicago_dem_df = chicago_mayoral[["BVAP_pct", "HVAP_pct", "WVAP_pct", "GARCIA_G15_pct"]]

#map household salary shares to the precinct level support for Garcia.
chicago_salary_df = chicago_mayoral[salary_cols + ["GARCIA_G15_pct"]]

#print(chicago_salary_df.head())

#Keplermapper cannot read an input dataframe, so we turn our dataframes into numpy arrays.
chicago_dem_data = np.array(chicago_dem_df)
chicago_salary_data = np.array(chicago_salary_df)

#double-check shape of data
#print(chicago_dem_data.shape)
#print(chicago_salary_data.shape)

#choose the column's number to project our data to.Here, the data is projected to the precinct level support for Chuy Garcia, so we want to get the index of that column out of the dataframe.
dem_col_num = chicago_dem_df.columns.get_loc("GARCIA_G15_pct")
salary_col_num = chicago_salary_df.columns.get_loc("GARCIA_G15_pct")

#So, we want to take our variable of interest (the election outcome), and fit a mapper object to this data.
chicago_dem_mapper = km.KeplerMapper(verbose=1)
chicago_dem_projected = chicago_dem_mapper.fit_transform(chicago_dem_data, projection = [dem_col_num], scaler=None)

chicago_salary_mapper = km.KeplerMapper(verbose=1)
chicago_salary_projected = chicago_salary_mapper.fit_transform(chicago_salary_data, projection = [salary_col_num], scaler=None)

print(chicago_dem_projected.shape)
#choose number of cubes and percent of overlap the cubes should have with eachother.
n_cubes = 40
p_overlap = .2
chicago_dem_graph = chicago_dem_mapper.map(chicago_dem_projected, chicago_dem_data, cover = km.Cover(n_cubes=n_cubes, perc_overlap=p_overlap))
chicago_salary_graph = chicago_salary_mapper.map(chicago_salary_projected, chicago_salary_data, cover = km.Cover(n_cubes=n_cubes, perc_overlap=p_overlap))

#use the Kepler Mapper visualization tool to see what our graph looks like.
chicago_dem_mapper.visualize(chicago_dem_graph, path_html = "chicago_demographics_garcia.html",
    title="Demographic Data for Chicago", color_values=chicago_dem_data[:,2], color_function_name='WVAP_pct', X = chicago_dem_data, X_names = ["BVAP_pct", "HVAP_pct", "WVAP_pct", "GARCIA_G15_pct"])

print(chicago_dem_data)

chicago_salary_mapper.visualize(chicago_salary_graph, path_html = "chicago_salary_garcia.html",
                                title="Salary Data for Chicago",
                                color_values=chicago_salary_data[:, chicago_salary_df.columns.get_loc('200K_MORE_pct')],
                                color_function_name='200K_MORE_pct', X=chicago_salary_data,
                                X_names=salary_cols + ["GARCIA_G15_pct"])

list(chicago_dem_graph["nodes"].keys())[:10]

#once you've created the mapper object, you can get specific cluster information.
cluster_info = chicago_dem_mapper.data_from_cluster_id('cube33_cluster1', chicago_dem_graph, chicago_dem_data)

print(len(chicago_dem_graph['nodes'])) #['cube33_cluster1']
cluster_df = pd.DataFrame(cluster_info, columns=chicago_dem_df.columns)

chicago_mayoral[["WVAP_pct", "BVAP_pct", "GARCIA_G15_pct", "EMAN_G15_pct"]][chicago_mayoral.index.isin(chicago_dem_graph['nodes']['cube8_cluster0'])]

#print(cluster_df.head())
# function here to plot the averages of a list of desired values within each cluster vs. another column of interest.
def graph_cluster_means(graph, columns, col_of_interest):
  colors = ["red", "coral", "bisque", "orange", "yellow", "goldenrod", "olivedrab", "springgreen", "steelblue"]
  votes = []
  cluster_means = [[] for col in range(len(columns))]
  fig, ax = plt.subplots(1,1, figsize=(7, 5))
  for cluster in graph['nodes'].keys():
    cluster_indices = graph['nodes'][cluster]
    subset = chicago_mayoral[chicago_mayoral.index.isin(cluster_indices)]
    votes.append(np.mean(subset[col_of_interest]))
    for i, col in enumerate(columns):
      cluster_means[i].append(np.mean(subset[col]))
  for i, cluster in enumerate(cluster_means):
    plt.scatter(cluster, votes, color=colors[i], label = columns[i])
  plt.xlabel("Demographic share")
  plt.ylabel(f"{col_of_interest}")
  plt.legend()
  plt.show()
graph_cluster_means(chicago_salary_graph, ["HVAP_pct", "BVAP_pct", "WVAP_pct"], "GARCIA_G15_pct")

# Import the Chicago precinct shapefile:
precinct_geodf = gpd.read_file("/Users/emariedelanuez/Chicago-Mayoral-Mapper-Map-Viz-Fork/keppler_mapper_data/PRECINCTS_2012.zip")
#shapefile_zip = files.upload()
#filename = list(shapefile_zip.keys())[0]
#!unzip $filename
chicago_mayoral[(chicago_mayoral["ward"]== 15) & (chicago_mayoral["precinct"]== 6)]
precinct_geodf[(precinct_geodf["WARD"]== 15) & (precinct_geodf["PRECINCT"]== 6)]

idxs = list()

for node in chicago_dem_graph["nodes"].values():
  idxs = idxs + list(node)

print(2069 - len(set(idxs)))

#The following function generates an interactive map using the folium library with layers corresponding to some set of precincts
### expects a dict of the form "layer_name": [...node_idxs...], ...
def generate_layered_map(layer_dict, default_open):
  map = folium.Map()

  for layer_name, precinct_idxs in layer_dict.items():
    precincts = chicago_mayoral[chicago_mayoral.index.isin(precinct_idxs)]
    precinct_ids = precincts["full_text"].apply(lambda n : str(n).rjust(5, "0"))

    geodf_clusters = precinct_geodf[precinct_geodf["FULL_TEXT"].isin(precinct_ids)]
    map = geodf_clusters.explore(m = map, name=layer_name, show=default_open)

  folium.LayerControl().add_to(map)
  map.fit_bounds(map.get_bounds(), padding=(20, 20))

  return map

generate_layered_map(chicago_dem_graph['nodes'], False)

#use networkx to get more interesting layers out of the topological information contained in the graph, i.e. in the example below, the connected components of the graph
conn_comp_layers = dict()
comp_sizes = list()

G = km.to_nx(chicago_dem_graph)

for C in sorted(nx.connected_components(G), key=len, reverse=True):
    subgraph = nx.subgraph(G, C)

    comp_precincts = list()

    for name, data in subgraph.nodes.items():
        comp_precincts = comp_precincts + data['membership']

    comp_size = len(list(subgraph.nodes))
    comp_sizes = comp_sizes + [comp_size]

    count = comp_sizes.count(comp_size)

    if count > 1:
        layer_name = ("%i cluster component (%i)" % (comp_size, count - 1))
    else:
        layer_name = ("%i cluster component" % (comp_size))

    conn_comp_layers[layer_name] = comp_precincts

generate_layered_map(conn_comp_layers, True)

#see more interesting geographic clustering, we can try to identify my topologically interesting subsets of vertices.

cycle_layers = dict()
cycles = nx.cycle_basis(G)
cut_points = list(nx.articulation_points(G))

for cycle in cycles:
    cycle_graph = nx.subgraph(G, cycle)
    join_verts = list(set(cut_points) & set(cycle_graph))

    intersection_precincts = G.nodes[join_verts[0]]["membership"] + G.nodes[join_verts[1]]["membership"]

    paths = list(nx.all_simple_paths(cycle_graph, join_verts[0], join_verts[1]))
    paths_as_precincts = list()

    for path in paths:
        path = list(set(path) - set(join_verts))
        path_precincts = list()

        path_n = 0

        for node in path:
            path_precincts += G.nodes[node]["membership"]
            if node.split("_")[1] == "cluster1": path_n = 1

        paths_as_precincts.insert(path_n, path_precincts)

    layer_name_prefix = join_verts[0].split("_")[0] + " - " + join_verts[1].split("_")[0] + " cycle "

    for n, path in enumerate(paths_as_precincts):
        layer = dict()
        name = layer_name_prefix + "path " + str(n)

        cycle_layers[name] = path

    intersection_layer = layer_name_prefix + "branch/merge points"
    cycle_layers[intersection_layer] = intersection_precincts

generate_layered_map(cycle_layers, False)