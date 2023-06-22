from abc import ABC, abstractmethod
from types import MappingProxyType
import sys

import math
import numpy as np
from math import e
import copy

from matplotlib import pyplot as plt
from scipy.special import expit as logistic

import source.bloodflowmodel.transmissibility as transmissibility
import source.bloodflowmodel.pressure_flow_solver as pressureflowsolver
import source.bloodflowmodel.build_system as buildsystem
import source.bloodflowmodel.rbc_velocity as rbc_velocity
from util_methods.util_plot import util_display_graph, graph_creation


def _logit(x):
    return math.log(x / (1 - x))


class DischargeHaematocrit(ABC):
    """
    Abstract base class for the implementations related to calculating the discharge haematocrit.
    """

    def __init__(self, PARAMETERS: MappingProxyType):
        """
        Constructor of DischargeHaematocrit
        :param PARAMETERS: Global simulation parameters stored in an immutable dictionary.
        :type PARAMETERS: MappingProxyType (basically an immutable dictionary).
        """
        self._PARAMETERS = PARAMETERS
        self.x_o_init = 1.12  # micrometers
        self.A_o_init = 15.47  # micrometers
        self.B_o_init = 8.13  # micrometers

    @abstractmethod
    def update_hd(self, flownetwork):
        """
        Abstract method to update the discharge haematocrit in flownetwork.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """

    @abstractmethod
    def qRCS(self, flownetwork, case, flow_a_par, flow_b_par, flow_c_par, flow_a_d, flow_b_d, flow_c_d, hemat_a_par, hemat_b_par, hemat_c_par, hemat_a_d, hemat_b_d, hemat_c_d):
        """
        check the RBC balance
        qRBC = q * Hdt

        Massbalances:
        q_p1 + q_p2 = q_d,
        qRBC_p1 + qRBC_p2 = qRBC_d1,
        """


class DischargeHaematocritNewtonian(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit without taking red blood cells into account.
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in the flow network with the zero vector for Newtonian flow.
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        flownetwork.hd = flownetwork.ht


class DischargeHaematocritVitroPries1992(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit based on the empirical in vitro functions by
    Pries, Neuhaus, Gaehtgens (1992).
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in flownetwork based on tube haematocrit and vessel diameter. The
        model==based on the empirical in vitro functions by Pries, Neuhaus, Gaehtgens (1992).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.35 * diameter_um) - 0.6 * np.exp(-0.01 * diameter_um)  # Eq. (9) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.99] = 0.99  # Bound x to values < 1. Equation in paper==only valid for x < 1.

        hd = -x_bound / (2 - 2 * x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2 * x_bound)) + ht / (1 - x_bound))  # Eq 10 in paper
        hd[x_tmp > 0.99] = ht[x_tmp > 0.99]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit


class DischargeHaematocritVitroPries2005(DischargeHaematocrit):
    """
    Class for updating the discharge haematocrit based on the empirical in vitro functions by
    Pries and Secomb (2005).
    """

    def update_hd(self, flownetwork):
        """
        Update the discharge haematocrit in flownetwork based on tube haematocrit and vessel diameter. The
        model==based on the empirical in vitro functions by Pries and Secomb (2005).
        :param flownetwork: flow network object
        :type flownetwork: source.flow_network.FlowNetwork
        """
        ht = flownetwork.ht  # Tube haematocrit
        diameter_um = 1.e6 * flownetwork.diameter  # Vessel diameter in micro meters

        x_tmp = 1. + 1.7 * np.exp(-0.415 * diameter_um) - 0.6 * np.exp(-0.011 * diameter_um)  # From Eq.(1) in paper
        x_bound = np.copy(x_tmp)
        x_bound[x_tmp > 0.99] = 0.  # Bound x to values < 1. Equation in paper==only valid for x < 1.

        hd = -x_bound / (2 - 2 * x_bound) + np.sqrt(
            np.square(x_bound / (2 - 2 * x_bound)) + ht / (1 - x_bound))
        hd[x_tmp > 0.99] = ht[x_tmp > 0.99]  # For very small and very large diameters: set ht=hd

        flownetwork.hd = hd  # Update discharge haematocrit


class DischargeHaematocritPries1990(DischargeHaematocrit):

    def qRCS(self, case, flow_a_par, flow_b_par, flow_c_par, flow_a_d, flow_b_d, flow_c_d, hemat_a_par, hemat_b_par, hemat_c_par, hemat_a_d, hemat_b_d, hemat_c_d, **kwargs):
        """
        check the RBC balance
        qRBC = q * Hdt

        Massbalances:
        q_p1 + q_p2 = q_d,
        qRBC_p1 + qRBC_p2 = qRBC_d1

        if there is one in/out use a value
        """

        tollerance = 1.00E-05
        RBCbalance = 1

        match case:

            # one parent and one daughter (-ø-) (1,1)
            case 1:
                if np.abs(flow_a_par - flow_a_d) <= tollerance:  # check same flow
                    qRBC_p = flow_a_par * hemat_a_par
                    qRBC_d = flow_a_d * hemat_a_d
                    if np.abs(qRBC_p - qRBC_d) <= tollerance:
                        RBCbalance = 0

            # one parent and two daughter (-<) (1,2)
            case 2:
                flow_in_daughters = flow_a_d + flow_b_d
                if np.abs(flow_a_par - flow_in_daughters) <= tollerance:  # check same flow
                    qRBC_a, qRBC_b = flow_a_d * hemat_a_d, flow_b_d * hemat_b_d
                    qRBC_p = flow_a_par * hemat_a_par
                    qRBC_d = qRBC_a + qRBC_b
                    if np.abs(qRBC_p - qRBC_d) <= tollerance:
                        RBCbalance = 0

            # two parent and one daughter (-E)
            case 3:
                flow_in_daughters = flow_a_d + flow_b_d + flow_c_d
                if np.abs(flow_a_par - flow_in_daughters) <= tollerance:  # check same flow
                    qRBC_a, qRBC_b, qRBC_c = flow_a_d * hemat_a_d, flow_b_d * hemat_b_d, flow_c_d * hemat_c_d
                    qRBC_p = flow_a_par * hemat_a_par
                    qRBC_d = qRBC_a + qRBC_b + qRBC_c
                    if np.abs(qRBC_p - qRBC_d) <= tollerance:
                        RBCbalance = 0

            # three parents and one daughter (>ø-)
            case 4:
                flow_in_parents = flow_a_par + flow_b_par
                if np.abs(flow_in_parents - flow_a_d) <= tollerance:  # check same flow
                    qRBC_a, qRBC_b = flow_a_par * hemat_a_par, flow_b_par * hemat_b_par
                    qRBC_d = flow_a_d * hemat_a_d
                    qRBC_p = qRBC_a + qRBC_b
                    if np.abs(qRBC_p - qRBC_d) <= tollerance:
                        RBCbalance = 0

            # (3ø-)
            case 5:
                flow_in_parents = flow_a_par + flow_b_par + flow_c_par
                if np.abs(flow_in_parents - flow_a_d) <= tollerance:  # check same flow
                    qRBC_a_p, qRBC_b_p, qRBC_c_p = flow_a_par * hemat_a_par, flow_b_par * hemat_b_par, flow_c_par * hemat_c_par
                    qRBC_d = flow_a_d * hemat_a_d
                    qRBC_p = qRBC_a_p + qRBC_b_p + qRBC_c_p
                    if np.abs(qRBC_p - qRBC_d) <= tollerance:
                        RBCbalance = 0

        if RBCbalance == 1:
            print("NOT BALANCED")
        return RBCbalance

    def hematocrit_1_1(self, hemat_parent, flow_parent, flow_daughter, rbc_balance):
        hematocrit = hemat_parent
        rbc_balance += self.qRCS(1, flow_parent, None, None, flow_daughter, None, None, hemat_parent, None, None, hematocrit, None, None)

        return hematocrit, rbc_balance

    def hematocrit_1_3(self, flow_parent, hemat_parent, flow_daughter_a, flow_daughter_b, flow_daughter_c, rbc_balance, fractional_trifurc_RBCs, fractional_trifurc_blood):
        hematocrit_a, hematocrit_b, hematocrit_c = hemat_parent, hemat_parent, hemat_parent

        rbc_balance += self.qRCS(3, flow_parent, None, None, flow_daughter_a, flow_daughter_b, flow_daughter_c, hemat_parent, None, None, hematocrit_a, hematocrit_b, hematocrit_c)

        # to print
        qRBC_parent = flow_parent * hemat_parent
        fractional_trifurc_RBCs.extend([(flow_daughter_a * hematocrit_a) / qRBC_parent, (flow_daughter_b * hematocrit_b) / qRBC_parent, (flow_daughter_c * hematocrit_c)
                                        / qRBC_parent])
        fractional_trifurc_blood.extend([flow_daughter_a / flow_parent, flow_daughter_b / flow_parent, flow_daughter_c / flow_parent])

        return hematocrit_a, hematocrit_b, hematocrit_c, rbc_balance

    def hematocrit_2_1(self, flow_parent_a, flow_parent_b, flow_daughter, hemat_parent_a, hemat_parent_b, rbc_balance):
        hematocrit = ((flow_parent_a * hemat_parent_a) + (flow_parent_b * hemat_parent_b)) / (flow_parent_a + flow_parent_b)

        rbc_balance += self.qRCS(4, flow_parent_a, flow_parent_b, None, flow_daughter, None, None, hemat_parent_a, hemat_parent_b, None, hematocrit, None, None)
        return hematocrit, rbc_balance

    def hematocrit_2_2_aux(self, hemat_a, hemat_b, flow_a, flow_b, diameter_a, diameter_b):
        diameter_parent = (diameter_a + diameter_b) / 2
        flow_parent = flow_a + flow_b
        hemat_parent = ((flow_a * hemat_a) + (flow_b * hemat_b)) / (flow_a + flow_b)

        return diameter_parent, flow_parent, hemat_parent

    def hematocrit_1_2(self, flow_parent, hemat_parent, flow_daughter_a, flow_daughter_b, rbc_balance, diameter_parent, diameter_a, diameter_b, fractional_a_qRBCs,
                       fractional_b_qRBCs, fractional_a_blood, fractional_b_blood, hemat_parent_plot):

        hematocrit_a, hematocrit_b = self.get_erythrocyte_fraction(hemat_parent,
                                                                   diameter_parent, diameter_a, diameter_b,
                                                                   flow_parent, flow_daughter_a, flow_daughter_b,
                                                                   fractional_a_qRBCs, fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                                                   hemat_parent_plot)
        rbc_balance += self.qRCS(2, flow_parent, None, None, flow_daughter_a, flow_daughter_b, None, hemat_parent, None, None, hematocrit_a, hematocrit_b, None)

        return hematocrit_a, hematocrit_b, rbc_balance

    def hematocrit_3_1(self, flow_parent_a, flow_parent_b, flow_parent_c, flow_daughter, hemat_parent_a, hemat_parent_b, hemat_parent_c, rbc_balance):
        """
        Computation of the daughter vessel hematocrit in case of 3 inflow and one outflows
        """
        # computation of the hematocrit for the daughter vessel
        hematocrit = ((flow_parent_a * hemat_parent_a) + (flow_parent_b * hemat_parent_b) + (flow_parent_c * hemat_parent_c)) / (flow_parent_a + flow_parent_b + flow_parent_c)

        rbc_balance += self.qRCS(5, flow_parent_a, flow_parent_b, flow_parent_c, flow_daughter, None, None, hemat_parent_a, hemat_parent_b, hemat_parent_c, hematocrit, None, None)

        return hematocrit, rbc_balance

    def non_dimentional_param(self, hemat_par, diam_par, diam_a, diam_b):
        diam_a, diam_b, diam_par = diam_a * 1E6, diam_b * 1E6, diam_par * 1E6

        x_0 = self.x_o_init * (1 - hemat_par) / diam_par

        A = (-self.A_o_init) * ((pow(diam_a, 2) - pow(diam_b, 2)) / (pow(diam_a, 2) + pow(diam_b, 2))) * (
                1 - hemat_par) / diam_par

        B = 1 + (self.B_o_init * (1 - hemat_par) / diam_par)

        return x_0, A, B

    def get_erythrocyte_fraction(self, hemat_par, diam_par, diam_a, diam_b,
                                 flow_parent, flow_a, flow_b,
                                 fractional_a_qRBCs, fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                 hemat_parent_plot):
        """
        to calculate the fraction of erythrocyte that goes in each daugheter vessel
        """
        fractional_flow_a = flow_a / (flow_a + flow_b)
        fractional_flow_b = flow_b / (flow_a + flow_b)

        x_0, A, B = self.non_dimentional_param(hemat_par, diam_par, diam_a, diam_b)

        qRBCp = hemat_par * flow_parent

        if fractional_flow_a <= x_0:
            fractional_qRBCa, fractional_qRBCb = 0, 1

        elif x_0 < fractional_flow_a < (1 - x_0):
            internal_logit = (fractional_flow_a - x_0) / (1 - x_0)
            logit_result = A + B * _logit(internal_logit)
            fractional_qRBCa = (pow(e, logit_result) / (1 + pow(e, logit_result)))
            fractional_qRBCb = 1 - fractional_qRBCa

        elif fractional_flow_a >= (1 - x_0):
            fractional_qRBCa, fractional_qRBCb = 1, 0

        # to avoid Nan
        if flow_a < 1E-36:
            hemat_a = 0
            hemat_b = (fractional_qRBCb * qRBCp) / flow_b
        elif flow_b < 1E-36:
            hemat_a = (fractional_qRBCa * qRBCp) / flow_a
            hemat_b = 0
        else:
            hemat_a = (fractional_qRBCa * qRBCp) / flow_a
            hemat_b = (fractional_qRBCb * qRBCp) / flow_b

        threshold = 0.99
        if hemat_b >= threshold:
            hemat_surplus = hemat_b - threshold
            fractional_RBCs_suprlus = (hemat_surplus * flow_b) / qRBCp

            fractional_qRBCb = fractional_qRBCb - fractional_RBCs_suprlus
            fractional_qRBCa = fractional_qRBCa + fractional_RBCs_suprlus

            hemat_a = (fractional_qRBCa * qRBCp) / flow_a
            hemat_b = (fractional_qRBCb * qRBCp) / flow_b
            # for plot

        elif hemat_a >= threshold:
            hemat_surplus = hemat_a - threshold
            fractional_RBCs_suprlus = (hemat_surplus * flow_a) / qRBCp

            # for plot
            fractional_qRBCb = fractional_qRBCb + fractional_RBCs_suprlus
            fractional_qRBCa = fractional_qRBCa - fractional_RBCs_suprlus
            hemat_a = (fractional_qRBCa * qRBCp) / flow_a
            hemat_b = (fractional_qRBCb * qRBCp) / flow_b

        fractional_a_qRBCs.append(fractional_qRBCa)
        fractional_b_qRBCs.append(fractional_qRBCb)
        # blood
        fractional_a_blood.append(fractional_flow_a)
        fractional_b_blood.append(fractional_flow_b)

        hemat_parent_plot.append(hemat_par)

        if hemat_a >= 1 or hemat_b >= 1:
            sys.exit()
        return hemat_a, hemat_b

    def update_hd(self, flownetwork):
        """

        """
        flownetwork.boundary_hematocrit = self._PARAMETERS["boundary_hematocrit"]
        flownetwork.fractional_a_qRBCs, flownetwork.fractional_b_qRBCs = [], []
        flownetwork.fractional_a_blood, flownetwork.fractional_b_blood = [], []
        flownetwork.fractional_trifurc_RBCs, flownetwork.fractional_trifurc_blood = [], []
        flownetwork.hemat_parent_plot = []
        fractional_trifurc_RBCs, fractional_trifurc_blood = flownetwork.fractional_trifurc_RBCs, flownetwork.fractional_trifurc_blood
        fractional_a_qRBCs = flownetwork.fractional_a_qRBCs
        fractional_b_qRBCs = flownetwork.fractional_b_qRBCs
        fractional_a_blood = flownetwork.fractional_a_blood
        fractional_b_blood = flownetwork.fractional_b_blood
        hemat_parent_plot = flownetwork.hemat_parent_plot
        # diameter
        diameter = copy.copy(flownetwork.diameter)
        # solved by the system
        pressure = copy.copy(flownetwork.pressure)
        # number of nodes
        nr_of_vs = copy.copy(flownetwork.nr_of_vs)
        # edge list
        edge_list = copy.copy(flownetwork.edge_list)
        # base of arrays
        pressure_node = np.zeros((nr_of_vs, 2))

        rbc_balance = 0

        if pressure is None:
            flownetwork.hd = copy.copy(flownetwork.ht)
        else:
            flow = np.abs(copy.copy(flownetwork.flow_rate))

            # [pressure][node] in a single array
            for pres in range(0, nr_of_vs):
                pressure_node[pres] = np.array([pressure[pres], pres])

            # ordered in base of pressure [pressure][node]
            pressure_node = pressure_node[pressure_node[:, 0].argsort()[::-1]]
            ordered_node = pressure_node[:, 1]
            # iterate over the nodes, ordered by pressure
            for node in pressure_node:
                # position of the node connected
                edge_connected_position = []
                # node connceted
                node_connected = []
                edge_daughter = []
                # position of the parent edge
                parent_edge = []
                edge_connected = []

                # create the list of the edge connected with that node
                for edge in range(0, len(edge_list)):
                    # save the position of the edge in the edge_list
                    if edge_list[edge][0] == node[1] or edge_list[edge][1] == node[1]:
                        edge_connected_position.append(edge)
                        edge_connected.append(edge_list[edge])
                        # all node connected
                        if edge_list[edge][0] == node[1]:
                            node_connected.append(edge_list[edge][1])
                        else:
                            node_connected.append(edge_list[edge][0])

                    # find which is the parent node and the daughters one
                for on in ordered_node:
                    # nodo nella lista oridnata è il nodo considerato, non si trova nella lista connected
                    # ho trovato dove sia nella lista ordinata
                    if on == node[1]:
                        for element in edge_connected_position:
                            if not np.isin(element, parent_edge):
                                edge_daughter.append(element)
                        break
                    elif np.isin(on, edge_connected):
                        # connected node è nella lista dei connessi ne ho trovato uno nella lista quindi è un
                        # parent lo tolgo dalla lista dei nodi connessi e lo inserisco nella lista di nodi
                        # parent
                        for i in range(0, (len(edge_connected))):
                            if np.isin(on, edge_connected[i]):
                                # positizione del parent edge
                                parent_edge.append(edge_connected_position[i])

                # DAUGHTERS
                # Check witch type of daughters (bifurcation)
                match len(edge_daughter):
                    case 1:  # one daughter (ø-)
                        single_daughter = edge_daughter[0]
                        n_daughter = 1
                    case 2:  # bifurcation (ø<)
                        daughter_a, daughter_b = edge_daughter[0], edge_daughter[1]
                        n_daughter = 2
                    case 3:  # triforcation
                        daughter_a, daughter_b, daughter_c = edge_daughter[0], edge_daughter[1], edge_daughter[2]
                        n_daughter = 3
                    case _:  # no daughters
                        n_daughter = 0

                # PARENTS
                # Check witch type of parents
                match len(parent_edge):
                    case 1:  # single parent (-ø)
                        parent = parent_edge[0]
                        n_parent = 1
                    case 2:  # two parents (>ø)
                        parent_a, parent_b = parent_edge[0], parent_edge[1]
                        n_parent = 2
                    case 3:  # triforcation
                        parent_a, parent_b, parent_c = parent_edge[0], parent_edge[1], parent_edge[2]
                        n_parent = 3
                    case _:
                        n_parent = 0

                # check it the node is a boundary node
                if np.isin(node[1], flownetwork.boundary_vs):
                    # check the flow between parents and daughters
                    match n_parent, n_daughter:
                        # grade = 2
                        # (-ø-)
                        # inflow ghost
                        case 0, 1:
                            # compute hematocrit and RBC balance
                            flownetwork.hd[single_daughter] = self.hematocrit_1_1(flownetwork.boundary_hematocrit[np.where(flownetwork.boundary_vs == node[1])[0][0]],
                                                                                  flow[single_daughter], flow[single_daughter], rbc_balance)
                        # (-ø-)
                        # outflows ghost
                        case 1, 0:
                            pass

                        # grade = 3
                        # (-ø<)
                        # inflow ghost
                        case 0, 2:
                            # diameter ghost parent: average of the parent diameter
                            diameter_parent = (diameter[daughter_a] + diameter[daughter_b]) / 2
                            # flow ghost parent: absolute sum of flow in daughter vessels
                            flow_parent = flow[daughter_a] + flow[daughter_b]
                            # hematocrit ghost parent: value of boundary hematocrit of that node
                            hemat_parent = flownetwork.boundary_hematocrit[np.where(flownetwork.boundary_vs == node[1])[0][0]]
                            # compute hematocrit for daughter vessel and check RBCs value
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], rbc_balance = self.hematocrit_1_2(flow_parent, hemat_parent, flow[daughter_a], flow[daughter_b],
                                                                                                                      rbc_balance, diameter_parent, diameter[daughter_a],
                                                                                                                      diameter[daughter_b],
                                                                                                                      fractional_a_qRBCs,
                                                                                                                      fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                                                                                                      hemat_parent_plot)

                        # (>ø-)
                        # outflow ghost
                        case 2, 0:
                            pass

                        case 1, 1:
                            # (>ø-)
                            # ghost INflow
                            if flownetwork.flow_rate[parent] < flownetwork.flow_rate[single_daughter]:
                                # flow ghost parent: surplus flow between the daughter and parent vessel
                                flow_ghost = flownetwork.flow_rate[single_daughter] - flownetwork.flow_rate[parent]
                                # hematocrit ghost parent: value of boundary hematocrit of that node
                                hematocrit_ghost = flownetwork.boundary_hematocrit[np.where(flownetwork.boundary_vs == node[1])[0][0]]
                                # compute hematocrit for daughter vessel and check RBCs value
                                flownetwork.hd[single_daughter], rbc_balance = self.hematocrit_2_1(flow[parent], flow_ghost, flow[single_daughter], flownetwork.hd[parent],
                                                                                                   hematocrit_ghost, rbc_balance)
                            # (-ø<)
                            # ghost OUTflow
                            else:
                                # diameter ghost daughter: same as the other daughter
                                diameter_ghost = diameter[single_daughter]
                                # flow ghost daughter: surplus flow between the parent and daughter vessel
                                flow_ghost = flownetwork.flow_rate[parent] - flownetwork.flow_rate[single_daughter]
                                # compute hematocrit for daughter vessel and check RBCs value
                                flownetwork.hd[single_daughter], hemat_ghost_daughter, rbc_balance = self.hematocrit_1_2(flow[parent], flownetwork.hd[parent],
                                                                                                                         flow[single_daughter],
                                                                                                                         flow_ghost,
                                                                                                                         rbc_balance, diameter[parent], diameter[single_daughter],
                                                                                                                         diameter_ghost,
                                                                                                                         fractional_a_qRBCs, fractional_b_qRBCs, fractional_a_blood,
                                                                                                                         fractional_b_blood, hemat_parent_plot)

                        # grade = 4
                        case 2, 1:
                            flow_parents = flow[parent_a] + flow[parent_b]
                            # (3ø-)
                            # INflow ghost
                            if flow[single_daughter] > flow_parents:
                                # flow ghost parent: surplus flow between the parents and daughter vessel
                                flow_parent_ghost = flow[single_daughter] - (flow[parent_a] + flow[parent_b])
                                # hematocrit ghost parent: value of boundary hematocrit of that node
                                hematocrit_ghost = flownetwork.boundary_hematocrit[np.where(flownetwork.boundary_vs == node[1])[0][0]]
                                # compute hematocrit for daughter vessel and check RBCs value
                                flownetwork.hd[single_daughter], rbc_balance = self.hematocrit_3_1(flow[parent_a], flow[parent_b], flow_parent_ghost,
                                                                                                   flow[single_daughter], flownetwork.hd[parent_a],
                                                                                                   flownetwork.hd[parent_b], hematocrit_ghost, rbc_balance)

                            # (>ø<)
                            # OUTflow ghost
                            else:
                                # flow ghost daughter: surplus flow between the parents and daughter vessel
                                flow_daughter_ghost = flow_parents - flow[single_daughter]
                                # combine parent value to consider them as one
                                diameter_parent, flow_parent, hemat_parent = self.hematocrit_2_2_aux(flownetwork.hd[parent_a], flownetwork.hd[parent_b], flow[parent_a],
                                                                                                     flow[parent_b], diameter[parent_a], diameter[parent_b])
                                # compute hematocrit for daughter vessel and check RBCs value
                                flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], rbc_balance = self.hematocrit_1_2(flow_parent, hemat_parent, flow[daughter_a],
                                                                                                                          flow_daughter_ghost,
                                                                                                                          rbc_balance, diameter_parent, diameter[daughter_a],
                                                                                                                          diameter[daughter_a], fractional_a_qRBCs,
                                                                                                                          fractional_b_qRBCs, fractional_a_blood,
                                                                                                                          fractional_b_blood,
                                                                                                                          hemat_parent_plot)

                        # (3ø-)
                        # outflow ghost
                        case 3, 0:
                            pass
                        # (-øE)
                        # inflow ghost
                        case 0, 3:
                            # hematocrit ghost parent: value of boundary hematocrit of that node
                            hematocrit_ghost = flownetwork.boundary_hematocrit[np.where(flownetwork.boundary_vs == node[1])[0][0]]
                            # flow ghost parent: absolute sum of the daughters vessel
                            flow_parent_ghost = flow[daughter_a] + flow[daughter_b] + flow[daughter_c]
                            # compute hematocrit for daughter vessel and check RBCs value
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], flownetwork.hd[daughter_c], rbc_balance = self.hematocrit_1_3(flow_parent_ghost,
                                                                                                                                                  hematocrit_ghost,
                                                                                                                                                  flow[daughter_a],
                                                                                                                                                  flow[daughter_b],
                                                                                                                                                  flow[daughter_c], rbc_balance,
                                                                                                                                                  fractional_trifurc_RBCs,
                                                                                                                                                  fractional_trifurc_blood)

                        case 1, 2:
                            # to understand if it is inflow or outflow ghost case
                            flow_daughter = flow[daughter_a] + flow[daughter_b]
                            # (-øE)
                            # OUTflow ghost
                            if flownetwork.flow_rate[parent] > flow_daughter:
                                # flow ghost daughter: surplus flow between parent and daughters vessel
                                flow_ghost_daughter = flow[parent] - flow_daughter
                                # compute hematocrit for daughter vessel and check RBCs value
                                flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], hematocrit_ghost_daughter, rbc_balance = self.hematocrit_1_3(flow[parent],
                                                                                                                                                     flownetwork.hd[parent],
                                                                                                                                                     flow[daughter_a],
                                                                                                                                                     flow[daughter_b],
                                                                                                                                                     flow_ghost_daughter,
                                                                                                                                                     rbc_balance,
                                                                                                                                                     fractional_trifurc_RBCs,
                                                                                                                                                     fractional_trifurc_blood)

                            # (>ø<)
                            # INflow ghost
                            else:
                                # flow ghost daughter: sum of flows of the daughters vessel
                                flow_ghost = (flow[daughter_a] + flow[daughter_b]) - flow[parent]
                                # combine parent value to consider them as one
                                diameter_parent, flow_parent, hemat_parent = self.hematocrit_2_2_aux(flownetwork.hd[parent], flownetwork.boundary_hematocrit[
                                    np.where(flownetwork.boundary_vs == node[1])[0][0]], flow[parent], flow_ghost, diameter[daughter_a], diameter[daughter_a])
                                # compute phase separation and chek RBC balance
                                flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], rbc_balance = self.hematocrit_1_2(flow_parent, hemat_parent, flow[daughter_a],
                                                                                                                          flow[daughter_b],
                                                                                                                          rbc_balance, diameter_parent, diameter[daughter_a],
                                                                                                                          diameter[daughter_b], fractional_a_qRBCs,
                                                                                                                          fractional_b_qRBCs, fractional_a_blood,
                                                                                                                          fractional_b_blood,
                                                                                                                          hemat_parent_plot)

                # NOT a boundary node
                else:
                    # match cases to understand in which case we are
                    match n_parent, n_daughter:
                        # (-ø-)
                        case 1, 1:
                            # compute hematocrit and check RBC balance
                            flownetwork.hd[single_daughter], rbc_balance = self.hematocrit_1_1(flownetwork.hd[parent], flow[parent], flow[single_daughter], rbc_balance)

                        # (-<)
                        case 1, 2:
                            # compute phase separation and check RBC balance
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], rbc_balance = self.hematocrit_1_2(flow[parent], flownetwork.hd[parent], flow[daughter_a],
                                                                                                                      flow[daughter_b],
                                                                                                                      rbc_balance, diameter[parent], diameter[daughter_a],
                                                                                                                      diameter[daughter_b], fractional_a_qRBCs,
                                                                                                                      fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                                                                                                      hemat_parent_plot)

                        # (-øE)
                        case 1, 3:
                            # compute hematocrit and check RBC balance
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], flownetwork.hd[daughter_c], rbc_balance = self.hematocrit_1_3(flow[parent],
                                                                                                                                                  flownetwork.hd[parent],
                                                                                                                                                  flow[daughter_a],
                                                                                                                                                  flow[daughter_b],
                                                                                                                                                  flow[daughter_c], rbc_balance,
                                                                                                                                                  fractional_trifurc_RBCs,
                                                                                                                                                  fractional_trifurc_blood)

                        # (>ø-)
                        case 2, 1:
                            # compute hematocrit and check RBC balance
                            flownetwork.hd[single_daughter], rbc_balance = self.hematocrit_2_1(flow[parent_a], flow[parent_b], flow[single_daughter], flownetwork.hd[parent_a],
                                                                                               flownetwork.hd[parent_b], rbc_balance)

                        # (>ø<)
                        case 2, 2:

                            # combine parent value to consider them as one
                            diameter_parent, flow_parent, hemat_parent = self.hematocrit_2_2_aux(flownetwork.hd[parent_a], flownetwork.hd[parent_b], flow[parent_a], flow[
                                parent_b], diameter[daughter_a], diameter[daughter_b])
                            # compute phase separation and chek RBC balance
                            flownetwork.hd[daughter_a], flownetwork.hd[daughter_b], rbc_balance = self.hematocrit_1_2(flow_parent, hemat_parent, flow[daughter_a],
                                                                                                                      flow[daughter_b],
                                                                                                                      rbc_balance, diameter_parent, diameter[daughter_a],
                                                                                                                      diameter[daughter_b], fractional_a_qRBCs,
                                                                                                                      fractional_b_qRBCs, fractional_a_blood, fractional_b_blood,
                                                                                                                      hemat_parent_plot)

                        # (∃ø-)
                        case 3, 1:
                            # combine parent value to consider them as one, compute hematocrit and RBCs balance
                            flownetwork.hd[single_daughter], rbc_balance = self.hematocrit_3_1(flow[parent_a], flow[parent_b], flow[parent_c], flow[single_daughter],
                                                                                               flownetwork.hd[parent_a], flownetwork.hd[parent_b], flownetwork.hd[parent_c],
                                                                                               rbc_balance)

                        case 0, _:
                            print("Node " + str(node) + " has no parent")

            print("Check RBCs balance: ...")
            if rbc_balance > 0:
                sys.exit("Check RBCs balance: FAIL -->", rbc_balance)
            else:
                print("Check RBCs balance: DONE")
            # util_display_graph(graph_creation(flownetwork), flownetwork.iteration, self._PARAMETERS, flownetwork)
