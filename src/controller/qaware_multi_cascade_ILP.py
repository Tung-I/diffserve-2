import pandas as pd
import numpy as np 
import gurobipy as gp
from gurobipy import GRB
import time
from collections import defaultdict


# model_comb_thres_config = [((2, 3), 0.6), ((0, 2, 3), (0.0, 0.5, 0.5)), ((2, 3), 0.4), ((0, 2, 3), (0.1, 0.5, 0.4)), ((0, 2, 3), (0.0, 0.7, 0.3)), ((2, 3), 0.3), ((1, 2, 3), (0.1, 0.6, 0.3)), ((2, 3), 0.2), ((0, 2, 3), (0.0, 0.8, 0.2)), ((0, 2, 3), (0.1, 0.7, 0.2)), ((1, 2, 3), (0.1, 0.8, 0.1)), ((0, 2, 3), (0.1, 0.8, 0.1)), ((2, 3), 0.1), ((1, 2, 3), (0.2, 0.7, 0.1)), ((0, 1, 2), (0.0, 0.1, 0.9)), ((1, 2, 3), (0.1, 0.9, 0.0)), ((1, 2), 0.9), ((2, 3), 0.0), ((0, 2, 3), (0.1, 0.9, 0.0)), ((1, 2), 0.8), ((1, 2, 3), (0.2, 0.8, 0.0)), ((0, 2, 3), (0.2, 0.8, 0.0)), ((0, 2), 0.8), ((1, 2), 0.7), ((0, 1, 2), (0.1, 0.3, 0.6)), ((1, 2), 0.6), ((0, 1, 2), (0.2, 0.2, 0.6)), ((1, 2, 3), (0.4, 0.6, 0.0)), ((0, 1, 2), (0.2, 0.3, 0.5)), ((1, 2, 3), (0.5, 0.5, 0.0)), ((1, 2), 0.5), ((1, 2), 0.4), ((0, 1, 2), (0.2, 0.4, 0.4)), ((1, 2, 3), (0.6, 0.4, 0.0)), ((0, 1, 2), (0.1, 0.6, 0.3)), ((1, 2), 0.3), ((1, 2, 3), (0.7, 0.3, 0.0)), ((0, 1, 2), (0.2, 0.6, 0.2)), ((0, 1, 2), (0.3, 0.5, 0.2)), ((1, 2, 3), (0.8, 0.2, 0.0)), ((1, 2), 0.2), ((0, 1, 2), (0.1, 0.8, 0.1)), ((0, 1, 2), (0.3, 0.6, 0.1)), ((1, 2, 3), (0.9, 0.1, 0.0)), ((1, 2), 0.1), ((0, 1), 0.8), ((0, 1, 3), (0.2, 0.8, 0.0)), ((0, 1, 3), (0.3, 0.7, 0.0)), ((0, 1, 2), (0.3, 0.7, 0.0)), ((0, 1), 0.7), ((0, 1), 0.6), ((1, 2, 3), (1.0, 0.0, 0.0)), ((1, 2), 0.0), ((0, 1, 3), (0.5, 0.5, 0.0)), ((0, 1), 0.5), ((0, 1, 3), (0.6, 0.4, 0.0)), ((0, 1), 0.4), ((0, 1), 0.3), ((0, 1, 3), (0.7, 0.3, 0.0)), ((0, 1, 2), (0.8, 0.2, 0.0)), ((0, 1), 0.2), ((0, 1, 3), (0.8, 0.2, 0.0)), ((0, 1, 2), (0.9, 0.1, 0.0)), ((0, 1), 0.1), ((0, 1, 3), (0.9, 0.1, 0.0)), ((0, 1, 3), (1.0, 0.0, 0.0)), ((0, 1), 0.0)]
# model_fid_config = [19.05, 19.12, 19.19, 19.28, 19.43, 19.47, 19.55, 19.6, 19.66, 19.74, 19.89, 19.97, 20.0, 20.21, 20.26, 20.31, 20.36, 20.4, 20.41, 20.43, 20.46, 20.53, 20.61, 20.63, 20.81, 20.94, 21.01, 21.13, 21.34, 21.38, 21.43, 21.77, 21.9, 21.95, 22.17, 22.43, 22.5, 23.01, 23.16, 23.17, 23.34, 23.85, 23.87, 24.14, 24.18, 24.7, 24.71, 24.82, 24.86, 24.88, 24.99, 25.15, 25.16, 25.44, 25.47, 25.83, 26.06, 26.57, 26.74, 27.55, 27.68, 27.76, 28.48, 28.57, 28.79, 29.75, 29.76]
model_comb_thres_config = [((2.0, 3.0), 0.2, 0.5, 0.6), ((2.0, 3.0), 0.5, 0.0, 0.5), ((2.0, 3.0), 0.4, 0.0, 0.4), ((2.0, 3.0), 0.3, 0.0, 0.3), ((2.0, 3.0), 0.2, 0.0, 0.2), ((2.0, 3.0), 0.1, 0.0, 0.1), ((1.0, 2.0), 0.8, 0.6, 0.92), ((1.0, 2.0), 0.8, 0.4, 0.88), ((1.0, 2.0), 0.6, 0.6, 0.84), ((1.0, 2.0), 0.8, 0.2, 0.84), ((1.0, 2.0), 0.7, 0.4, 0.82), ((1.0, 2.0), 0.8, 0.1, 0.82), ((1.0, 2.0), 0.6, 0.4, 0.76), ((1.0, 2.0), 0.7, 0.2, 0.76), ((1.0, 2.0), 0.5, 0.4, 0.7), ((1.0, 2.0), 0.6, 0.2, 0.68), ((1.0, 2.0), 0.4, 0.4, 0.64), ((1.0, 2.0), 0.6, 0.1, 0.64), ((1.0, 2.0), 0.5, 0.2, 0.6), ((1.0, 2.0), 0.4, 0.3, 0.58), ((1.0, 2.0), 0.5, 0.1, 0.55), ((1.0, 2.0), 0.4, 0.2, 0.52), ((1.0, 2.0), 0.1, 0.4, 0.46), ((1.0, 2.0), 0.4, 0.1, 0.46), ((1.0, 2.0), 0.3, 0.2, 0.44), ((1.0, 2.0), 0.0, 0.4, 0.4), ((1.0, 2.0), 0.1, 0.3, 0.37), ((1.0, 2.0), 0.2, 0.2, 0.36), ((1.0, 2.0), 0.3, 0.1, 0.37), ((1.0, 2.0), 0.0, 0.3, 0.3), ((1.0, 2.0), 0.1, 0.2, 0.28), ((1.0, 2.0), 0.2, 0.1, 0.28), ((1.0, 2.0), 0.0, 0.2, 0.2), ((1.0, 2.0), 0.1, 0.1, 0.19), ((1.0, 2.0), 0.2, 0.0, 0.2), ((1.0, 2.0), 0.0, 0.1, 0.1), ((1.0, 2.0), 0.1, 0.0, 0.1), ((0.0, 1.0), 0.7, 0.1, 0.73), ((0.0, 1.0), 0.7, 0.0, 0.7), ((0.0, 1.0), 0.6, 0.1, 0.64), ((0.0, 1.0), 0.6, 0.0, 0.6), ((0.0, 1.0), 0.5, 0.1, 0.55), ((0.0, 1.0), 0.5, 0.0, 0.5), ((0.0, 1.0), 0.4, 0.1, 0.46), ((0.0, 1.0), 0.4, 0.0, 0.4), ((0.0, 1.0), 0.3, 0.0, 0.3), ((0.0, 1.0), 0.2, 0.0, 0.2), ((0.0, 1.0), 0.1, 0.0, 0.1), ((0.0, 1.0), 0.0, 0.0, 0.0)]
model_fid_config = [19.29, 19.3, 19.42, 19.52, 19.77, 20.16, 20.5, 20.52, 20.54, 20.55, 20.56, 20.59, 20.62, 20.65, 20.72, 20.79, 20.88, 20.93, 20.98, 21.1, 21.2, 21.26, 21.52, 21.6, 21.68, 21.79, 21.98, 22.03, 22.16, 22.37, 22.45, 22.6, 22.93, 23.18, 23.45, 23.85, 24.3, 24.88, 24.91, 25.0, 25.08, 25.35, 25.56, 25.66, 25.98, 26.54, 27.43, 28.51, 29.75]

def create_lookup_table(data_list, fid_list, total_models=4):
    lookup_data = []

    for idx, (models, router_thres, conf_thres, thresholds) in enumerate(data_list):
        # Convert model_combination into a binary list representation (e.g., [0,1,0,1] for models (1,3))
        model_binary = [1 if i in models else 0 for i in range(total_models)]

        # Initialize route ratios with zeros
        route_ratio = [0.0] * total_models

        num_models = len(models)

        if num_models == 2:
            # Single threshold (float) provided
            print(f"models[0]: {models[0]}, models[1]: {models[1]}, thresholds: {thresholds}")
            route_ratio[int(models[0])] = 1.0  # First model always gets 100% of queries
            route_ratio[int(models[1])] = round(thresholds, 2)  # Second model gets threshold % of queries

        elif num_models == 3:
            # Three thresholds provided (tuple of 3 floats)
            t1, t2, t3 = thresholds
            route_ratio[models[0]] = round(t1 + t2 + t3, 1)  # Queries entering the first model
            route_ratio[models[1]] = round(t2 + t3, 1)  # Queries forwarded to the second model
            route_ratio[models[2]] = round(t3, 1)  # Queries forwarded to the third model

        elif num_models == 4:
            # Four thresholds provided (tuple of 4 floats)
            t1, t2, t3, t4 = thresholds
            route_ratio[models[0]] = round(t1 + t2 + t3 + t4, 1)  # Queries entering the first model
            route_ratio[models[1]] = round(t2 + t3 + t4, 1)  # Queries forwarded to the second model
            route_ratio[models[2]] = round(t3 + t4, 1)  # Queries forwarded to the third model
            route_ratio[models[3]] = round(t4, 1)  # Queries forwarded to the fourth model

        else:
            raise ValueError("Only 2, 3, or 4-model cascades are supported.")

        # Store the processed data
        lookup_data.append({
            "model_combination": model_binary,
            "router_thres": router_thres,
            "conf_thres": conf_thres,
            "route_ratio": route_ratio,
            "fid": fid_list[idx]
        })

    # Convert to DataFrame
    lookup_table = pd.DataFrame(lookup_data)
    lookup_table["model_combination"] = lookup_table["model_combination"].apply(tuple)
    return lookup_table


def solve_milp_loop(total_servers, sysDemand, latencySLO, lookup_table, model_latency_table,
                    demand_per_model, queue_length_per_model):
    '''
    Solves an MILP using NumPy arrays for:
    - Model combination selection
    - Routing ratio computations
    - Batch size and hardware allocation
    - Minimizing FID while ensuring latency constraints
    '''
    
    if sysDemand == 0:
        return None
    
    best_fid = float("inf")
    best_solution = None
    ilp_overhead_total = 0

    # Estimate queuing delay per model using Little's Law
    queuing_delay = defaultdict(float)
    queue_safety_factor = 1.2
    for model_id in range(len(lookup_table["model_combination"].iloc[0])):
        demand = demand_per_model.get(model_id, 0)
        queue_len = queue_length_per_model.get(model_id, 0)
        queuing_delay[model_id] = 0 if demand == 0 else queue_safety_factor * queue_len / demand
    
    grouped_table = lookup_table.groupby("model_combination")
    
    # iterate over model combinations
    for md_comb, sub_table in grouped_table:
        model_combination = np.array(md_comb)
        model_indices = np.where(model_combination == 1)[0] # extract active model indices
        
        route_ratios_list = np.vstack(sub_table["route_ratio"].to_numpy())
        fid_values = sub_table["fid"].to_numpy()
        
        # create MILP model
        m = gp.Model('Multi-model cascade MILP')
        m.setParam("LogToConsole", 0)
        m.setParam("Threads", 12)
        
        # Parameters
        allowed_batch_sizes = np.array([1, 2, 4, 8, 16, 32])
        slo = latencySLO  # Assume latencySLO is in seconds
        
        # Define Decision Variables as NumPy Arrays
        x = np.array([m.addVar(vtype=GRB.INTEGER, name=f'x_{i}') for i in range(len(model_combination))])
        b = np.array([
            np.array([m.addVar(vtype=GRB.BINARY, name=f'b_{i}_{batch}') for batch in allowed_batch_sizes])
            for i in range(len(model_combination))
        ])
        thr_ind = np.array([m.addVar(vtype=GRB.BINARY, name=f'threshold_{k}') for k in range(len(fid_values))])
        fid_score = m.addVar(vtype=GRB.CONTINUOUS, name='FID')
        
        # Constraints
        for i in range(len(x)):
            m.addConstr(x[i] >= 0) # Ensure non-negative device allocation
        
        for i in range(len(model_combination)):
            m.addConstr(b[i].sum() == 1)  # One batch size per model
        
        # Ensure only one route ratio is selected
        m.addConstr(thr_ind.sum() == 1)
        
        # Resource constraint: total allocated devices must not exceed available resources
        m.addConstr(x.sum() <= total_servers - 1)
        
        # Compute throughput as batch_size / latency
        model_latency_values = np.array([
            [model_latency_table.get((i, batch), 1) for batch in allowed_batch_sizes]
            for i in range(len(model_combination))
        ])
        model_throughput = allowed_batch_sizes / model_latency_values # Vectorized Throughput computation
        
        # Vectorized Throughput Constraint
        for i in range(len(model_combination)):
            m.addConstr(
                np.sum(b[i,j] * model_throughput[i,j] for j in range(len(allowed_batch_sizes))) * x[i]
                >= sysDemand * np.sum(thr_ind[k] * route_ratios_list[k, i] for k in range(len(fid_values)))
            )
        # m.addConstr((b * model_throughput).sum(axis=1) * x >= sysDemand * (thr_ind @ route_ratios_list))

        # Select FID using threshold indicator
        m.addConstr(fid_score == np.sum(thr_ind[k] * fid_values[k] for k in range(len(fid_values))))  
        # m.addConstr(fid_score == np.sum(thr_ind * fid_values)) # select the minimum FID from the active combination
        
        # Compute End-to-End Latency (Vectorized)
        selected_latency = np.array([
            np.sum(b[i,j] * model_latency_values[i,j] for j in range(len(allowed_batch_sizes))) + queuing_delay[i]
            for i in range(len(model_combination))
        ])
        # # this computes latency with routing-probability-weighted average
        # total_latency = np.sum(thr_ind[k] * route_ratios_list[k, i] * selected_latency[i] for k in range(len(fid_values))
        #                     for i in range(len(model_combination)))
        # this compute the sum of latency of all active models in the cascade
        total_latency = np.sum([
            selected_latency[i] for i in range(len(model_combination)) if model_combination[i] == 1
        ])
        # selected_latency = (b * model_latency_values).sum(axis=1)  # Select correct batch latency for each model
        # total_latency = np.dot(thr_ind @ route_ratios_list, selected_latency)
        
        # Latency Constraint
        m.addConstr(total_latency <= slo)
        
        # Optimization Objective: Minimize FID
        m.setObjective(fid_score, GRB.MINIMIZE)
        
        # Solve the optimization
        start_time = time.time()
        m.optimize()
        end_time = time.time()
        ilp_overhead = end_time - start_time
        print(f'Time to solve MILP for model combination {model_indices}: {ilp_overhead:.4f} seconds')
        ilp_overhead_total += ilp_overhead
        
        # Store the best solution
        if m.Status == GRB.OPTIMAL:
            solution_fid = fid_score.X
            selected_k = np.argmax([thr_ind[k].X for k in range(len(fid_values))])

            if solution_fid < best_fid:
                best_fid = solution_fid
                best_solution = {
                    "models": model_indices.tolist(),
                    "router_thres": sub_table.iloc[selected_k]["router_thres"],
                    "conf_thres": sub_table.iloc[selected_k]["conf_thres"],
                    "route_ratio": {
                        i: np.sum(thr_ind[k].X * route_ratios_list[k,i] for k in range(len(fid_values)))
                        for i in model_indices
                    },
                    "batch_sizes": {i: allowed_batch_sizes[np.argmax([b[i][j].X for j in range(len(allowed_batch_sizes))])] 
                                    for i in model_indices},
                    "device_allocation": {i: int(x[i].X) for i in model_indices},
                    "FID": solution_fid
                }
    
    print(f'Time to solve MILP: {ilp_overhead_total:.4f} seconds')
    if best_solution:
        print(f'Best Solution: {best_solution}')
    else:
        print("No feasible solution found.")
        
    return best_solution
        

def solve_milp_no_loop(total_servers, sysDemand, latencySLO, lookup_table, model_latency_table):
    '''
    Solves an MILP **without looping**, using:
    - Model combination selection via `z_c`
    - Routing ratio selection via `thr_{c,k}`
    - Batch size and device allocation
    - Minimization of FID while ensuring latency and throughput constraints
    '''
    if sysDemand == 0:
        return None

    best_solution = None
    
    # Extract model combinations and threshold configurations
    model_combinations = np.array(lookup_table["model_combination"].tolist())
    route_ratios_list = np.vstack(lookup_table["route_ratio"].to_numpy())  # 2D NumPy array
    fid_values = lookup_table["fid"].to_numpy()  # FID scores for each (combination, threshold)
    num_combinations = len(model_combinations)
    '''
    TODO: group by model_combination first, then get num_combinations
    '''
    
    # Create MILP model
    m = gp.Model('Multi-Model Cascade MILP')
    m.setParam("LogToConsole", 0)
    m.setParam("Threads", 12)
    
    # Parameters
    total_models = model_combinations.shape[1]  # Number of models (columns in model_combinations)
    allowed_batch_sizes = np.array([1, 2, 4, 8, 16, 32])  # Possible batch sizes
    slo = latencySLO  # Latency constraint
    
    # Define Decision Variables
    x = np.array([m.addVar(vtype=GRB.INTEGER, name=f'x_{i}') for i in range(total_models)])
    b = np.array([
        np.array([m.addVar(vtype=GRB.BINARY, name=f'b_{i}_{batch}') for batch in allowed_batch_sizes])
        for i in range(total_models)
    ])
    thr = np.array([[m.addVar(vtype=GRB.BINARY, name=f'thr_{c}_{k}') for k in range(len(fid_values))] 
                    for c in range(num_combinations)])
    z = np.array([m.addVar(vtype=GRB.BINARY, name=f'z_{c}') for c in range(num_combinations)])
    fid_score = m.addVar(vtype=GRB.CONTINUOUS, name='FID')
    
    # Auxiliary Variables for Linearizing Latency Computation
    L = np.array([
        [np.array([m.addVar(vtype=GRB.CONTINUOUS, name=f'L_{c}_{i}_{k}') 
                   for k in range(len(fid_values))]) 
         for i in range(total_models)]
        for c in range(num_combinations)
    ])
    
    # Constraints
    for i in range(len(x)):
        m.addConstr(x[i] >= 0) # Ensure non-negative device allocation
    
    for i in range(total_models):
        m.addConstr(b[i].sum() == 1)  # One batch size per model
        
    # Model Combination Selection: Only One Combination Can Be Used
    m.addConstr(z.sum() == 1)
    
    # Threshold Selection: One per Selected Combination
    for c in range(num_combinations):
        m.addConstr(thr[c].sum() == z[c])
    
    # Resource Constraint: Total Devices
    m.addConstr(x.sum() <= total_servers - 1)
    
    # Compute throughput as batch_size / latency
    model_latency_values = np.array([
        [model_latency_table.get((i, batch), 1) for batch in allowed_batch_sizes]
        for i in range(total_models)
    ])
    model_throughput = allowed_batch_sizes / model_latency_values # Vectorized Throughput computation
    
    # Vectorized throughput constraint
    for i in range(total_models):
        m.addConstr(
            np.sum(b[i,j] * model_throughput[i,j] for j in range(len(allowed_batch_sizes))) * x[i]
            >= sysDemand * np.sum(z[c] * np.sum(thr[c,k] * route_ratios_list[k,i] for k in range(len(fid_values)))
                            for c in range(num_combinations))
        )
    # m.addConstr((b * model_throughput).sum(axis=1) * x >= sysDemand * (z @ route_ratios_list))

    # Set FID Using Selected Model Combination and Threshold
    m.addConstr(fid_score == np.sum(z[c] * sum(thr[c,k] * fid_values[k] for k in range(len(fid_values)))
                                   for c in range(num_combinations)))
    # m.addConstr(fid_score == np.sum(thr * fid_values))  # Selects the minimum FID from the active combination
    
    # Linearized Latency Computation
    selected_latency = np.array([
        np.sum(b[i, j] * model_latency_values[i, j] for j in range(len(allowed_batch_sizes))) 
        for i in range(total_models)
    ])
    
    for c in range(num_combinations):
        for i in range(total_models):
            for k in range(len(fid_values)):
                m.addConstr(L[c, i, k] >= thr[c, k] * route_ratios_list[k, i] * selected_latency[i])
                m.addConstr(L[c, i, k] <= thr[c, k] * route_ratios_list[k, i] * (selected_latency[i] + 1e-6))

    total_latency = np.sum(z[c] * np.sum(L[c, i, k] for k in range(len(fid_values)) for i in range(total_models))
                        for c in range(num_combinations))

    # # Compute End-to-End Latency (Vectorized)
    # selected_latency = [
    #     np.sum(b[i,j] * model_latency_values[i,j] for j in range(len(allowed_batch_sizes)))
    #     for i in range(total_models)
    # ]
    # total_latency = np.sum(z[c] * sum(thr[c,k] * route_ratios_list[k,i] * selected_latency[i] for k in range(len(fid_values)))
    #                       for c in range(num_combinations)
    #                       for i in range(total_models))
    # # selected_latency = (b * model_latency_values).sum(axis=1)  # Select correct batch latency for each model
    # # total_latency = np.dot(z @ route_ratios_list, selected_latency)  # Weighted latency

    # Latency Constraint
    m.addConstr(total_latency <= slo)
    
    # Objective: Minimize FID
    m.setObjective(fid_score, GRB.MINIMIZE)
    
    # Solve the Optimization
    start_time = time.time()
    m.optimize()
    end_time = time.time()
    ilp_overhead = end_time - start_time
    print(f'Time to solve MILP: {ilp_overhead:.4f} seconds')
    
    # Store the Best Solution
    selected_c = np.argmax([z[c].X for c in range(num_combinations)])
    selected_k = np.argmax([thr[selected_c][k].X for k in range(len(fid_values))])
    selected_model_indices = np.where(model_combinations[selected_c] == 1)[0]
    
    if m.Status == GRB.OPTIMAL:
        best_solution = {
            "models": selected_model_indices.tolist(),
            "router_thres": lookup_table.iloc[selected_k]["router_thres"],
            "conf_thres": lookup_table.iloc[selected_k]["conf_thres"],
            "route_ratio": {
                i: np.sum(
                    z[c].X * np.sum(thr[c,k].X * route_ratios_list[k,i] for k in range(len(fid_values)))
                    for c in range(num_combinations)
                )
                for i in selected_model_indices
            },
            "batch_sizes": {i: allowed_batch_sizes[np.argmax([b[i][j].X for j in range(len(allowed_batch_sizes))])] 
                            for i in selected_model_indices},
            "device_allocation": {i: int(x[i].X) for i in selected_model_indices},
            "FID": fid_score.X
        }
        
    if best_solution:
        print(f'Best Solution: {best_solution}')
    else:
        print("No feasible solution found.")

    return best_solution


def solve_proteus_milp(model_latency_values, total_workers, slo, ewma_demand, fid_weighting, 
                       demand_per_model, queue_length_per_model):
    """
    Proteus-style ILP allocator that:
    - maximizes accuracy (weighted query volume),
    - satisfies latency, EWMA demand, and worker constraints.

    Parameters:
    - model_latency_values: dict of (model, batch) -> latency in second
    - fid_weighting: dict of model -> accuracy score (higher = better)
    """
    if ewma_demand == 0:
        return None
    
    allowed_batch_sizes = [1, 2, 4, 8, 16, 32]
    model_names = sorted(set([model for (model, b) in model_latency_values]))
    num_models = len(model_names)

    queuing_delay = defaultdict(float)
    queue_safety_factor = 1.2
    for model_name in model_names:
        demand = demand_per_model.get(model_name, 0)
        queue_len = queue_length_per_model.get(model_name, 0)
        queuing_delay[model_name] = 0 if demand == 0 else queue_safety_factor * queue_len / demand

    m = gp.Model("proteus_accuracy_allocator")
    m.setParam('OutputFlag', 0)

    # Variables
    b = m.addVars(num_models, len(allowed_batch_sizes), vtype=GRB.BINARY, name="b")
    w = m.addVars(num_models, vtype=GRB.INTEGER, name="w", lb=0)

    # One batch size per model
    for i in range(num_models):
        m.addConstr(gp.quicksum(b[i, j] for j in range(len(allowed_batch_sizes))) <= 1)

    for i in range(num_models):
        y_i = gp.quicksum(b[i, j] for j in range(len(allowed_batch_sizes)))  # 0 or 1
        m.addConstr(w[i] <= (total_workers - 1) * y_i)  # if no batch, w[i]=0
        m.addConstr(w[i] >= y_i) 
        # Total worker limit
        m.addConstr(gp.quicksum(w[i] for i in range(num_models)) <= total_workers-1)

    # Latency constraint
    for i, model in enumerate(model_names):
        for j, bs in enumerate(allowed_batch_sizes):
            latency = model_latency_values.get((model, bs), None) * (1+queuing_delay[model])
            if latency is not None:
                m.addConstr(b[i, j] * latency <= slo)

    # Throughput constraint to meet demand
    total_throughput = gp.quicksum(
        w[i] * b[i, j] * allowed_batch_sizes[j] / model_latency_values.get((model_names[i], allowed_batch_sizes[j]), 1)
        for i in range(num_models)
        for j in range(len(allowed_batch_sizes))
        if (model_names[i], allowed_batch_sizes[j]) in model_latency_values
    )
    m.addConstr(total_throughput >= ewma_demand)

    # Objective: maximize quality-weighted throughput
    # weighted_accuracy = gp.quicksum(
    #     fid_weighting.get(model_names[i], 1.0) *
    #     w[i] * b[i, j] * allowed_batch_sizes[j] /
    #     model_latency_values.get((model_names[i], allowed_batch_sizes[j]), 1)
    #     for i in range(num_models)
    #     for j in range(len(allowed_batch_sizes))
    #     if (model_names[i], allowed_batch_sizes[j]) in model_latency_values
    # )
    # m.setObjective(weighted_accuracy, GRB.MAXIMIZE)
    eps = 1e-9
    vals = [fid_weighting[m] for m in model_names]
    fmin, fmax = min(vals), max(vals)
    den = (fmax - fmin)

    if den < eps:
        # all equal: give equal weight
        inv_weight = {m: 1.0 for m in model_names}
    else:
        inv_weight = {m: (fmax - fid_weighting[m]) / (den + eps) for m in model_names}
    weighted_quality = gp.quicksum(inv_weight[model_names[i]] * w[i]
                               for i in range(num_models))
    m.setObjective(weighted_quality, GRB.MAXIMIZE)

    m.optimize()

    allocation = {
        "device_allocation": {model_names[i]: int(w[i].x) for i in range(num_models)},
        "batch_sizes": {model_names[i]: allowed_batch_sizes[j]
                        for i in range(num_models)
                        for j in range(len(allowed_batch_sizes)) if b[i, j].x > 0.5},
    }
    print(f'Best Solution: {allocation}')

    return allocation



if __name__=='__main__':
    lookup_table = create_lookup_table(model_comb_thres_config, model_fid_config)