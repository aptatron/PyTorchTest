import sqlite3
# import networkx as nx
import json
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
nodeset = []
edgeset = []
dataset_node = None
dataset_edge = None


def read(database_path):
    global dataset_node
    global dataset_edge
    print("in read")

    # connect to database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # retrieve data from node_table
    cursor.execute("SELECT * FROM node_table;")
    dataset_node = cursor.fetchall()
    print(dataset_node)

    cursor.execute("SELECT  * FROM edge_table;")
    dataset_edge = cursor.fetchall()

    conn.close()


def node(nodes, switch):
    global nodeset
    global dataset_node
    if switch == "1":
        # create list of nodes
        print(dataset_node)
        for row in dataset_node:
            new_node = {
                'id': row[0],
                'name': row[1],
                'description': row[2]
            }
            nodeset.append(new_node)
            print("new Node")
            print(new_node)
    elif switch == "2":
        # retrieve data from form
        data = nodes
        name = data.get('name')
        description = data.get('description')

        # create a list of nodes
        new_node = {
            'id': len(dataset_node) + 1,
            'name': name,
            'description': description
        }
        nodeset.append(new_node)
    else:
        # invalid switch value
        raise ValueError('Invalid switch value. Must be 1 or 2. switch =' + switch)
    # print(nodeset)


def edge(edges, switch):
    global edgeset
    global dataset_edge
    if switch == "1":
        # create a list of edges
        for row in dataset_edge:
            new_edge = {
                'source_id': row[0],
                'target_id': row[1],
                'weight': row[2]
            }
            edgeset.append(new_edge)
    elif switch == "2":
        # retrieve data from form
        data = edges
        source = data.get('source_id')
        target = data.get('target_id')
        weight = data.get('weight')

        # create a list of edges
        new_edge = {
            'source_id': source,
            'target_id': target,
            'weight': weight
        }
        edgeset.append(new_edge)
    else:
        # invalid switch value
        raise ValueError('Invalid switch value. Must be 1 or 2.')


def network():
    global nodeset
    global edgeset
    graph = {
        "nodes": nodeset,
        "edges": edgeset
    }
    with open('graph1.json', 'w') as f:
        json.dump(graph, f, indent=4)


def write_nodes(database_path):
    # connect to the database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # delete existing nodes
    cursor.execute("DELETE FROM node_table;")

    # insert new nodes
    for node in dataset_node:
        cursor.execute("INSERT INTO node_table (id, name, description) VALUES (?, ?, ?);", (node['id'], node['name'], node['description']))

    conn.commit()
    conn.close()


def write_edges(database_path):
    # connect to the database
    conn = sqlite3.connect(database_path)
    cursor = conn.cursor()

    # delete existing edges
    cursor.execute("DELETE FROM edge_table;")

    # insert new edges
    for edge in dataset_edge:
        cursor.execute("INSERT INTO edge_table (source_id, target_id, weight) VALUES (?, ?, ?) ", (edge['source_id'], edge['target_id'], edge['weight']))

    conn.commit()
    conn.close()


@app.route('/update_network/<my_param>', methods=['POST'])
def update_network(my_param):
    global nodeset
    global edgeset
    print("update network")
    if my_param == "1":
        read('/database')
    new_data = request.get_json()
    node(new_data, my_param)
    edge(new_data, my_param)
    network()
    if my_param == "2":
        write_nodes('/database')
        write_edges('/database')
    with open('graph1.json', 'r') as f:
        graph_data = json.load(f)
    print("here")
    print(graph_data)
    return jsonify(graph_data)


if __name__ == '__main__':
    # update_network(1)
    app.run()



