import gurobipy as gp
from gurobipy import GRB
from pulp import *
from utils import *
folder_path = "Data"

class PortfolioModelPulp(DataProcessor):
    def __init__(self, data_processor):
        super().__init__(folder_path) 
        self.__dict__.update(data_processor.__dict__)  # Inherit all variables from DataProcessor
        self.semaines_chosen=None
        self.scenario_chosen=None
        self.CA_expr=None
        self.CMO_expr=None
        self.CV_expr=None
        self.obj=None       
        self.list_obj = None
        self.constraints=[]

    def optimize_portfolio(self):
  
        try:
            m = LpProblem("portfolio", LpMaximize)
            choices = LpVariable.dicts("choice", [(i, j, t) for i in range(self.num_serre)
                                          for j in self.scenarios
                                          for t in self.month_to_week_indices[self.scenario_mois_dict[j]]], 
                               cat=LpBinary)

            for i in range(self.num_serre):
                m += lpSum(choices[(i, j, t)] for j in self.scenarios
                   for t in self.month_to_week_indices[self.scenario_mois_dict[j]]) == 1

            for i in range(self.num_sect):
                ref = self.secteur_serre_dict[i + 1][0]
                for j in self.secteur_serre_dict[i + 1]:
                    for k in self.scenarios:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[k]]:
                            m += choices[(j - 1, k, t)] == choices[(ref - 1, k, t)]

            for i in range(21):
                for t in self.month_to_week_indices[self.scenario_mois_dict[5]]:
                    m += choices[(i, 5, t)] == 0
                for t in self.month_to_week_indices[self.scenario_mois_dict[4]]:
                    m += choices[(i, 4, t)] == 0
                

            for i in range(self.num_serre):
                if i != 19 and i != 20:
                    for j in self.variety_scenario_dict["Clara"]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            m += choices[(i, j, t)] == 0
                    for j in self.variety_scenario_dict["LAURITA"]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            m += choices[(i, j, t)] == 0
                else:
                    if self.serre_sau_dict[i + 1] > 2.87:
                        for j in self.variety_scenario_dict["Clara"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m += choices[(i, j, t)] == 0
                        for j in self.variety_scenario_dict["LAURITA"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m += choices[(i, j, t)] == 0

            for s in range(1, 91):
                scenarios_time_dict = {}
                for j in self.scenarios:
                    scenarios_time_dict[j] = []
                    if j in [4, 5, 20]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + 38:
                                scenarios_time_dict[j].append(t)
                    else:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + self.scenario_duree_dict[j] + 1:
                                scenarios_time_dict[j].append(t)
                m += lpSum(choices[(i, k, t)] * (self.serre_sau_dict[i + 1] / self.scenario_vitesse[k]) *
                   self.prod_mat[self.scenarios.index(k), s - t - self.scenario_delai_dict[k] - 1]
                   for k in self.scenarios for t in scenarios_time_dict[k] if scenarios_time_dict[k] != []
                   for i in range(self.num_serre)) <= 7 * 600

            CA_expr = lpSum(choices[(i, j, t)] * self.prod[(i, j, t)]
                    for i in range(self.num_serre) for j in self.scenarios
                    for t in self.month_to_week_indices[self.scenario_mois_dict[j]])

            CMO_expr = lpSum(choices[(i, j, t)] * self.scenario_prod[j] * self.serre_sau_dict[i + 1] * self.scenario_cmo[j]
                     for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                     for i in range(self.num_serre))

            CV_expr = lpSum(choices[(i, j, t)] * self.serre_sau_dict[i + 1] * self.scenario_cv[j]
                    for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                    for i in range(self.num_serre))
            obj = lpSum(choices[(i, j, t)] * (self.prod[(i, j, t)]-self.scenario_prod[j] * self.serre_sau_dict[i + 1] * self.scenario_cmo[j]-self.serre_sau_dict[i + 1] * self.scenario_cv[j])
                for i in range(self.num_serre) for j in self.scenarios
                for t in self.month_to_week_indices[self.scenario_mois_dict[j]])

            m += obj
            m.solve()
            CA_expr=value(CA_expr)[0][0]
            CMO_expr=value(CMO_expr)
            CV_expr=value(CV_expr)
            self.CA_expr=CA_expr
            self.CMO_expr=CMO_expr
            self.CV_expr=CV_expr
            obj=CA_expr - CV_expr - CMO_expr
            self.obj=obj
            print("Value of CA_expr:",CA_expr )
            print("Value of CMO_expr:", CMO_expr)
            print("Value of CV_expr:", CV_expr)
            print("Obj:", obj)

            scenario_chosen, semaines_chosen = [], []
            if LpStatus[m.status] == "Optimal":
        # Get selected variables with value == 1
                list_var = [v.name for v in m.variables() if v.varValue == 1]
                sorted_data = sorted(list_var, key=lambda x: (int(x.split(',')[0].split('(')[-1]), int(x.split(',')[1].split('_')[-1]), int(x.split(',')[2].split('_')[-1].rstrip(')'))))

            for i in sorted_data:
                 scenario_chosen.append(int(i.split("_")[2][:-1]))
                 semaines_chosen.append(int(i.split("_")[3][:-1]))
            self.scenario_chosen=scenario_chosen
            self.semaines_chosen=semaines_chosen
            
            
        except Exception as e:
            print(e)


    def get_top_k(self, n):
        try:
            m = LpProblem("portfolio", LpMaximize)
            choices = LpVariable.dicts("choice", [(i, j, t) for i in range(self.num_serre)
                                      for j in self.scenarios
                                      for t in self.month_to_week_indices[self.scenario_mois_dict[j]]], 
                           cat=LpBinary)

            for i in range(self.num_serre):
                m += lpSum(choices[(i, j, t)] for j in self.scenarios
                   for t in self.month_to_week_indices[self.scenario_mois_dict[j]]) == 1

            for i in range(self.num_sect):
                ref = self.secteur_serre_dict[i + 1][0]
                for j in self.secteur_serre_dict[i + 1]:
                    for k in self.scenarios:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[k]]:
                            m += choices[(j - 1, k, t)] == choices[(ref - 1, k, t)]

            for i in range(21):
                for t in self.month_to_week_indices[self.scenario_mois_dict[5]]:
                    m += choices[(i, 5, t)] == 0
                for t in self.month_to_week_indices[self.scenario_mois_dict[4]]:
                    m += choices[(i, 4, t)] == 0

            for i in range(self.num_serre):
                if i != 19 and i != 20:
                    for j in self.variety_scenario_dict["Clara"]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            m += choices[(i, j, t)] == 0
                    for j in self.variety_scenario_dict["LAURITA"]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            m += choices[(i, j, t)] == 0
                else:
                    if self.serre_sau_dict[i + 1] > 2.87:
                        for j in self.variety_scenario_dict["Clara"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m += choices[(i, j, t)] == 0
                        for j in self.variety_scenario_dict["LAURITA"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m += choices[(i, j, t)] == 0

            for s in range(1, 91):
                scenarios_time_dict = {}
                for j in self.scenarios:
                    scenarios_time_dict[j] = []
                    if j in [4, 5, 20]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + 38:
                                scenarios_time_dict[j].append(t)
                    else:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + self.scenario_duree_dict[j] + 1:
                                scenarios_time_dict[j].append(t)
                m += lpSum(choices[(i, k, t)] * (self.serre_sau_dict[i + 1] / self.scenario_vitesse[k]) *
               self.prod_mat[self.scenarios.index(k), s - t - self.scenario_delai_dict[k] - 1]
                    for k in self.scenarios for t in scenarios_time_dict[k] if scenarios_time_dict[k] != []
               for i in range(self.num_serre)) <= 7 * 600

            
            
            CA_expr = lpSum(choices[(i, j, t)] * self.prod[(i, j, t)]
                    for i in range(self.num_serre) for j in self.scenarios
                    for t in self.month_to_week_indices[self.scenario_mois_dict[j]])

            CMO_expr = lpSum(choices[(i, j, t)] * self.scenario_prod[j] * self.serre_sau_dict[i + 1] * self.scenario_cmo[j]
                     for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                     for i in range(self.num_serre))

            CV_expr = lpSum(choices[(i, j, t)] * self.serre_sau_dict[i + 1] * self.scenario_cv[j]
                    for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                    for i in range(self.num_serre))
            obj = lpSum(choices[(i, j, t)] * (self.prod[(i, j, t)]-self.scenario_prod[j] * self.serre_sau_dict[i + 1] * self.scenario_cmo[j]-self.serre_sau_dict[i + 1] * self.scenario_cv[j])
                for i in range(self.num_serre) for j in self.scenarios
                for t in self.month_to_week_indices[self.scenario_mois_dict[j]])

            m.setObjective(obj)
            m.solve()
            CA=value(CA_expr)[0][0]
            CMO=value(CMO_expr)
            CV=value(CV_expr)
            obj_value=CA - CV - CMO            
            list_obj = [obj_value]
        
            for _ in range(n):
                m = LpProblem("portfolio", LpMaximize)
                choices = LpVariable.dicts("choice", [(i, j, t) for i in range(self.num_serre)
                                      for j in self.scenarios
                                      for t in self.month_to_week_indices[self.scenario_mois_dict[j]]], 
                           cat=LpBinary)

                for i in range(self.num_serre):
                    m += lpSum(choices[(i, j, t)] for j in self.scenarios
                    for t in self.month_to_week_indices[self.scenario_mois_dict[j]]) == 1

                for i in range(self.num_sect):
                    ref = self.secteur_serre_dict[i + 1][0]
                    for j in self.secteur_serre_dict[i + 1]:
                        for k in self.scenarios:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[k]]:
                                m += choices[(j - 1, k, t)] == choices[(ref - 1, k, t)]

                for i in range(21):
                    for t in self.month_to_week_indices[self.scenario_mois_dict[5]]:
                        m += choices[(i, 5, t)] == 0
                    for t in self.month_to_week_indices[self.scenario_mois_dict[4]]:
                        m += choices[(i, 4, t)] == 0

                for i in range(self.num_serre):
                    if i != 19 and i != 20:
                        for j in self.variety_scenario_dict["Clara"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m += choices[(i, j, t)] == 0
                        for j in self.variety_scenario_dict["LAURITA"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m += choices[(i, j, t)] == 0
                    else:
                        if self.serre_sau_dict[i + 1] > 2.87:
                            for j in self.variety_scenario_dict["Clara"]:
                                for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                    m += choices[(i, j, t)] == 0
                            for j in self.variety_scenario_dict["LAURITA"]:
                                for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                    m += choices[(i, j, t)] == 0

                for s in range(1, 91):
                    scenarios_time_dict = {}
                    for j in self.scenarios:
                        scenarios_time_dict[j] = []
                        if j in [4, 5, 20]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + 38:
                                    scenarios_time_dict[j].append(t)
                        else:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + self.scenario_duree_dict[j] + 1:
                                    scenarios_time_dict[j].append(t)
                    m += lpSum(choices[(i, k, t)] * (self.serre_sau_dict[i + 1] / self.scenario_vitesse[k]) *
               self.prod_mat[self.scenarios.index(k), s - t - self.scenario_delai_dict[k] - 1]
                        for k in self.scenarios for t in scenarios_time_dict[k] if scenarios_time_dict[k] != []
                for i in range(self.num_serre)) <= 7 * 600

            
            
                CA_expr = lpSum(choices[(i, j, t)] * self.prod[(i, j, t)]
                    for i in range(self.num_serre) for j in self.scenarios
                    for t in self.month_to_week_indices[self.scenario_mois_dict[j]])

                CMO_expr = lpSum(choices[(i, j, t)] * self.scenario_prod[j] * self.serre_sau_dict[i + 1] * self.scenario_cmo[j]
                     for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                     for i in range(self.num_serre))

                CV_expr = lpSum(choices[(i, j, t)] * self.serre_sau_dict[i + 1] * self.scenario_cv[j]
                    for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                    for i in range(self.num_serre))
                obj = lpSum(choices[(i, j, t)] * (self.prod[(i, j, t)]-self.scenario_prod[j] * self.serre_sau_dict[i + 1] * self.scenario_cmo[j]-self.serre_sau_dict[i + 1] * self.scenario_cv[j])
                for i in range(self.num_serre) for j in self.scenarios
                for t in self.month_to_week_indices[self.scenario_mois_dict[j]])
                m.addConstraint(obj <= list_obj[-1]-10)
                m.setObjective(obj)
                m.solve()
                CA=value(CA_expr)[0][0]
                CMO=value(CMO_expr)
                CV=value(CV_expr)
                obj_value=CA - CV - CMO
                list_obj.append(obj_value)
            
                
            self.list_obj=list_obj
        except Exception as e:
            print(e)
        
        
        

class PortfolioModelGurobi(DataProcessor):
    def __init__(self, data_processor):
        super().__init__(folder_path)  # Initialize the superclass (DataProcessor)
        self.__dict__.update(data_processor.__dict__)  # Inherit all variables from DataProcessor
        self.semaines_chosen=None
        self.scenario_chosen=None
        self.CA_expr=None
        self.CMO_expr=None
        self.CV_expr=None
        self.m=None
        self.choices=None
        self.list_obj=None
    def optimize_portfolio(self):
        try:
            m = gp.Model("portfolio")
            choices = {}
            for i in range(self.num_serre):
                for j in self.scenarios:
                    for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                        choices[(i, j, t)] = m.addVar(vtype=GRB.BINARY, name=f'choice_{i}_{j}_{t}')

            for i in range(self.num_serre):
                m.addConstr(gp.quicksum(choices[(i, j, t)] for j in self.scenarios 
                                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]) == 1, f"constraint_{i}")

            for i in range(self.num_sect):
                ref = self.secteur_serre_dict[i+1][0]
                for j in self.secteur_serre_dict[i+1]:
                    for k in self.scenarios:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[k]]:
                            m.addConstr(choices[(j-1,k,t)] == choices[(ref-1,k,t)], f'c_0_{j}_{k}_{t}')

            for i in range(21):
                for t in self.month_to_week_indices[self.scenario_mois_dict[5]]:
                    m.addConstr(choices[(i,5,t)] == 0)
                for t in self.month_to_week_indices[self.scenario_mois_dict[4]]:
                    m.addConstr(choices[(i,4,t)] == 0)

            for i in range(self.num_serre):
                if i!=19 and i!=20:
                    for j in self.variety_scenario_dict["Clara"]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            m.addConstr(choices[(i,j,t)] == 0)
                    for j in self.variety_scenario_dict["LAURITA"]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            m.addConstr(choices[(i,j,t)] == 0)
                else:
                    if self.serre_sau_dict[i+1] > 2.87:
                        for j in self.variety_scenario_dict["Clara"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m.addConstr(choices[(i,j,t)] == 0)
                        for j in self.variety_scenario_dict["LAURITA"]:
                            for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                                m.addConstr(choices[(i,j,t)] == 0)

            for s in range(1, 91):
                scenarios_time_dict = {}
                for j in self.scenarios:
                    scenarios_time_dict[j] = []
                    if j in [4, 5, 20]:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + 38:
                                scenarios_time_dict[j].append(t)
                    else:
                        for t in self.month_to_week_indices[self.scenario_mois_dict[j]]:
                            if t + self.scenario_delai_dict[j] < s and s < t + self.scenario_delai_dict[j] + self.scenario_duree_dict[j] + 1:
                                scenarios_time_dict[j].append(t)
                m.addConstr(gp.quicksum(choices[(i,k,t)]*(self.serre_sau_dict[i+1]/self.scenario_vitesse[k]) *
                                         self.prod_mat[self.scenarios.index(k),s-t-self.scenario_delai_dict[k]-1]
                                         for k in self.scenarios for t in scenarios_time_dict[k] if scenarios_time_dict[k]!=[]
                                         for i in range(self.num_serre)) <= 7*600, f'mo_{s}')

            CA_expr = gp.quicksum(choices[(i,j,t)] * self.prod[(i, j, t)] 
                                  for i in range(self.num_serre) for j in self.scenarios 
                                  for t in self.month_to_week_indices[self.scenario_mois_dict[j]])

            CMO_expr = gp.quicksum(choices[(i,j,t)] * self.scenario_prod[j] * self.serre_sau_dict[i+1] * self.scenario_cmo[j] 
                                   for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                                   for i in range(self.num_serre))

            CV_expr = gp.quicksum(choices[(i,j,t)] * self.serre_sau_dict[i+1] * self.scenario_cv[j] 
                                  for j in self.scenarios for t in self.month_to_week_indices[self.scenario_mois_dict[j]]
                                  for i in range(self.num_serre))

            m.update()
            m.setObjective((CA_expr - CV_expr - CMO_expr), GRB.MAXIMIZE)
            m.write('model.lp')
            m.optimize()
            self.CA_expr=CA_expr.getValue()
            self.CMO_expr=CMO_expr.getValue()
            self.CV_expr=CV_expr.getValue()

            print("Value of CA_expr:", CA_expr.getValue())
            print("Value of CMO_expr:", CMO_expr.getValue())
            print("Value of CV_expr:", CV_expr.getValue())
            print(f"Obj: {m.ObjVal:g}")
            # Your optimization code here, accessing variables inherited from DataProcessor
            scenario_chosen, semaines_chosen = [], []
            if m.status == GRB.Status.OPTIMAL:
                # Get selected variables with value == 1
                list_var = [v.varName for v in m.getVars() if v.x == 1]

            for i in list_var:
                scenario_chosen.append(int(i.split("_")[2]))
                semaines_chosen.append(int(i.split("_")[3]))
            self.semaines_chosen=semaines_chosen
            self.scenario_chosen=scenario_chosen
            
            

# Initialize a table to store chosen variable names
# Define the start date  
        except gp.GurobiError as e:
            print(f"Error code {e.errno}: {e}")
        except AttributeError:
            print("Encountered an attribute error")
    
    def get_top_k(self, n):
    # Initialize a list to store objective values
        
    
    # Read the MPS file and initialize the model
        m = gp.read('model.lp')
        choices = {v.varName: v for v in m.getVars()}
        m.optimize()
        list_obj = [m.ObjVal]
    # Print out the keys present in the choices dictionary
        
    
    # Loop for 'n' iterations
        chosen_variables_table = []
        for _ in range(n):
        # Optimize the model
        # Retrieve and store chosen variable names
            chosen_variables = [var_name for var_name, var in choices.items() if var.x == 1]
            chosen_variables_table.append(chosen_variables)
        
        # Add constraints to avoid repeating selections
            m.addConstr(gp.quicksum(choices[var_name] for var_name in chosen_variables) <= self.num_serre - 1)
        
        # Update and re-optimize the model with added constraints
            m.update()
            m.optimize()
        
        # Store the objective value
            list_obj.append(m.ObjVal)
        
    # Set the list of objective values
        self.list_obj = list_obj
    def robust_optimization(self):
        pass
