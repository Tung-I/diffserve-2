CAS_EXEC = 'multi' # modify config in controller.py, load_balancer.py, model.py, qaware_cascade_ILP.py ['sdturbo', 'sdxs', 'sdxlltn', 'multi']
DO_SIMULATE = False

CASCADE_MODEL_ORDER = ['sdxlltn', 'sd35turbo', 'sd35med', 'sd35large'] # models used for multi-level cascade

def get_model_order():
    return CASCADE_MODEL_ORDER

def get_cas_exec():
    return CAS_EXEC

def set_cas_exec(cascade):
    global CAS_EXEC
    CAS_EXEC = cascade
    
def get_do_simulate():
    return DO_SIMULATE

def set_do_simulate_true():
    global DO_SIMULATE
    DO_SIMULATE = True

def set_do_simulate_false():
    global DO_SIMULATE
    DO_SIMULATE = False
    
