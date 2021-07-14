#!/usr/bin/env python
import io
from itertools import chain

from io import BytesIO
import pyparsing as pp
import ast
import numpy as np


# TODO input and output state


import numpy as np

from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD, State


class NETReader(object):
    """
    Base class for reading network file in XMLBIF format.
    """

    def __init__(self, path=None, string=None, n_jobs=1):
        """
        Initialisation of XMLBIFReader object.

        Parameters
        ----------
        path : file or str
            File of XMLBIF data
            File of XMLBIF data

        string : str
            String of XMLBIF data

        Examples
        --------
        # xmlbif_test.xml is the file present in
        # http://www.cs.cmu.edu/~fgcozman/Research/InterchangeFormat/
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("net_test.net")
        >>> model = reader.get_model()
        """
        if path:
            self.network = self.getNetwork(open(path, "r"))
        elif string:
            self.network =  self.getNetwork(io.StringIO(string))
        else:
            raise ValueError("Must specify either path or string")
        self.network_name = "undefined"
        self.variables = self.get_variables()
        self.variable_parents = self.get_parents()
        self.edge_list = self.get_edges()
        self.state_names = self.get_states()
        self.variable_states = self.get_states()
        self.variable_CPD = self.get_values()

    def get_variables(self):
        """
        Returns list of variables of the network

        Examples
        --------
        >>> reader = NET.NETReader("net_test.net")
        >>> reader.get_variables()
        ['light-on', 'bowel-problem', 'dog-out', 'hear-bark', 'family-out']
        """
        return list(self.network.keys())

    def get_edges(self):
        """
        Returns the edges of the network

        Examples
        --------
        >>> reader = NET.NETReader("net_test.net")
        >>> reader.get_edges()
        [['family-out', 'light-on'],
         ['family-out', 'dog-out'],
         ['bowel-problem', 'dog-out'],
         ['dog-out', 'hear-bark']]
        """
        edge_list = [
            [value, key]
            for key in self.variable_parents
            for value in self.variable_parents[key]
        ]
        return edge_list

    def get_states(self):
        """
        Returns the states of variables present in the network

        Examples
        --------
        >>> reader = net.NETReader("net_test.net")
        >>> reader.get_states()
        {'bowel-problem': ['true', 'false'],
         'dog-out': ['true', 'false'],
         'family-out': ['true', 'false'],
         'hear-bark': ['true', 'false'],
         'light-on': ['true', 'false']}
        """
        states = {}
        for node in self.network.keys():
            state_string = self.network[node]["states"]
            state_string = state_string.replace("(", "")
            state_string = state_string.replace('"', "")
            state_string = state_string.replace(")", "")
            state_string = state_string.replace(";", "")
            states[node] = state_string.split()

        return states

    def get_parents(self):
        """
        Returns the parents of the variables present in the network

        Examples
        --------
        >>> reader = NET.NETReader("net_test.net")
        >>> reader.get_parents()
        {'bowel-problem': [],
         'dog-out': ['family-out', 'bowel-problem'],
         'family-out': [],
         'hear-bark': ['dog-out'],
         'light-on': ['family-out']}
        """
        parents = {}
        for node in self.network.keys():
            parents[node] = self.network[node]["parents"]

        return parents

    def get_values(self):
        """
        Returns the CPD of the variables present in the network

        Examples
        --------
        >>> reader = NET.NETReader("net_test.net")
        >>> reader.get_values()
        {'bowel-problem': array([[ 0.01],
                                 [ 0.99]]),
         'dog-out': array([[ 0.99,  0.01,  0.97,  0.03],
                           [ 0.9 ,  0.1 ,  0.3 ,  0.7 ]]),
         'family-out': array([[ 0.15],
                              [ 0.85]]),
         'hear-bark': array([[ 0.7 ,  0.3 ],
                             [ 0.01,  0.99]]),
         'light-on': array([[ 0.6 ,  0.4 ],
                            [ 0.05,  0.95]])}
        """
        cpd = {}
        for node in self.network.keys():
            cpd_string  = self.network[node]["cpd"]
            cpd_string = cpd_string.replace("(", "[")
            cpd_string = cpd_string.replace(")", "],")
            cpd_string = cpd_string.replace(" ", ", ")
            cpd_string = cpd_string.strip(",\n")
            if cpd_string == "":
                return []
            cpd_numpy = np.array(ast.literal_eval(cpd_string))
            if len(cpd_numpy.shape) == 1:
                cpd_numpy.resize((cpd_numpy.shape[0], 1))
            else:
                resize_dim = 1
                for i in range(0, len(cpd_numpy.shape)-1):
                    resize_dim *= cpd_numpy.shape[i]

                resize_tuple = (len(self.variable_states[node]), resize_dim)
                cpd_numpy = np.resize(cpd_numpy, resize_tuple)
            cpd[node] = cpd_numpy

        return cpd


    def get_model(self, state_name_type=str):
        """
        Returns a Bayesian Network instance from the file/string.

        Parameters
        ----------
        state_name_type: int, str, or bool (default: str)
            The data type to which to convert the state names of the variables.

        Returns
        -------
        BayesianNetwork instance: The read model.

        Examples
        --------
        >>> from pgmpy.readwrite import NETReader
        >>> reader = NETReader("net_test.net")
        >>> model = reader.get_model()
        """
        model = BayesianNetwork()
        model.add_nodes_from(self.variables)
        model.add_edges_from(self.edge_list)
        model.name = self.network_name

        tabular_cpds = []
        for var, values in self.variable_CPD.items():
            evidence_card = [
                len(self.variable_states[evidence_var])
                for evidence_var in self.variable_parents[var]
            ]
            cpd = TabularCPD(
                var,
                len(self.variable_states[var]),
                values,
                evidence=self.variable_parents[var],
                evidence_card=evidence_card,
                state_names={
                    var: list(map(state_name_type, self.state_names[var]))
                    for var in chain([var], self.variable_parents[var])
                },
            )
            tabular_cpds.append(cpd)

        model.add_cpds(*tabular_cpds)

        return model

    def getNetwork(self, file):
        nodes = {}
        while True:
            line = file.readline()
            if line == '':
                break
            words = line.split()
            if len(words) > 0:
                # get nodes
                if words[0] == "node":
                    id = words[1]
                    node = {}
                    if "{" in file.readline():
                        while True:
                            line = file.readline()
                            if "}" in line:
                                break
                            words = line.split("=")
                            node[words[0].strip()] = words[1].strip()
                    nodes[id] = node
                # get potentials
                if words[0] == "potential":
                    name = words[1].lstrip("(")
                    parents = []
                    for i in range(3, len(words)):
                        parents.append(words[i].replace(")", ""))
                    nodes[name]["parents"] = parents
                    if "{" in file.readline():
                        line = file.readline()
                        data = line.strip("data = ")
                        while True:
                            line = file.readline()
                            data += line.strip()
                            if "}" in line:
                                break
                        data = data.replace("}", "")
                        data = data.replace(";", "")
                        nodes[name]["cpd"] = data

        return nodes