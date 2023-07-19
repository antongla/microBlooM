import numpy as np

import source.flow_network as flow_network


class FlowBalance(object):

    def __init__(self, flownetwork: flow_network.FlowNetwork):

        self.flownetwork = flownetwork

    def _get_flow_balance(self):

        nr_of_vs = self.flownetwork.nr_of_vs
        nr_of_es = self.flownetwork.nr_of_es

        edge_list = self.flownetwork.edge_list

        flow_balance = np.zeros(nr_of_vs)
        flow_rate = self.flownetwork.flow_rate

        for eid in range(nr_of_es):
            flow_balance[edge_list[eid, 0]] += flow_rate[eid]
            flow_balance[edge_list[eid, 1]] -= flow_rate[eid]

        return flow_balance

    def check_flow_balance(self, tol=1.00E-05):
        nr_of_vs = self.flownetwork.nr_of_vs
        flow_rate = self.flownetwork.flow_rate
        boundary_vs = self.flownetwork.boundary_vs
        flow_balance = self._get_flow_balance()

        ref_flow = np.abs(flow_rate[boundary_vs[0]])
        tol_flow = tol * ref_flow

        is_inside_node = np.logical_not(np.in1d(np.arange(nr_of_vs), boundary_vs))
        local_balance = np.abs(flow_balance[is_inside_node])

        if self.flownetwork.tollerance is None:
            self.flownetwork.tollerance = np.mean(local_balance)
            print("tollerance :" + str(self.flownetwork.tollerance))


        else:

            is_locally_balanced = local_balance < tol_flow
            if False in np.unique(is_locally_balanced):
                import sys
                sys.exit("Is locally balanced: " + str(np.unique(is_locally_balanced)) + "(with tol " + str(tol_flow) + ")")

            balance_boundaries = flow_balance[boundary_vs]
            global_balance = np.abs(np.sum(balance_boundaries))
            is_globally_balanced = global_balance < tol_flow
            if not is_globally_balanced:
                import sys
                sys.exit("Is globally balanced: " + str(is_globally_balanced) + "(with tol " + str(tol_flow) + ")")

        return

    def tolerance(self):

        return 0
