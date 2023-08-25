import networkx as nx
import csv
import json
from flask import Flask, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def add_node_from_database(node_file, graph):
    with open(node_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        print(csv_reader)
        for row in csv_reader:
            node_id = row[0]
            name = row[1]
            description = row[2]
            graph.add_node(node_id, node=name, description=description)


def add_edge_from_database(edge_file, graph):
    with open(edge_file, 'r') as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            source = row[0]
            target = row[1]
            graph.add_edge(source, target)


@app.route('/add_node', methods=['POST'])
def add_node():
    node_file = './Nodes.csv'
    edge_file = './Edges.csv'
    G = nx.Graph()
    add_node_from_database(node_file, G)
    add_edge_from_database(edge_file, G)
    data = request.get_json()
    name = data.get('name')
    description = data.get('description')
    from networkx.readwrite import json_graph
    data = json_graph.node_link_data(G)
    with open('graph1.json', 'w') as f:
        json.dump(data, f, indent=4)



if __name__ == '__main__':
    app.run()









#
# G = nx.Graph()
#
#
# G.add_node(1, name='Node 1', description='Alzheimers')
# G.add_node(2, name='Node 2', description='resveratrol')
#
# G.add_edge(1, 2)
#
# app = Flask(__name__)
# CORS(app)
#
# nodes_list = list(G.nodes(data=True))
#
#
# if __name__ == '__main__':
#     app.run()
#
#
# # def update_visualization(node_list):
# #    nodes = []
# #     for node_id in node_list:
# #        node_data = G.nodes[node_id]
# #        node_dict = {'id': node_id, 'name': node_data['name'], 'description': node_data['description']}
# #         nodes.append(node_dict)
# #
# #     data = {'nodes': nodes}
# #
# #     return json.dumps(data)
#
#
# def add_node(name, description):
#     with open("graph1.json", "r") as f:
#         data = json.load(f)
#
#     new_node = {"name": name, "description": description, "id": len(data["nodes"]) + 1}
#
#     data["nodes"].append(new_node)
#
#     with open('graph1.json', 'w') as f:
#         json.dump(data, f, indent=4)
#
#
# @app.route('/add_node', methods=['POST'])
# def add_node_handler():
#     # Get the data from the request
#     name = request.json.get('name')
#     description = request.json.get('description')
#
#     # Call the add_node function
#     add_node(name, description)
#
#     # Return a response
#     return 'Node added successfully'
#
# from networkx.readwrite import json_graph
# data = json_graph.node_link_data(G)
# with open('graph1.json', 'w') as f:
#     json.dump(data, f, indent=4)
#
#
#
#
#


