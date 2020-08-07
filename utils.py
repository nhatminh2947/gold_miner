def policy_mapping(agent_id):
    # agent_id pattern training/opponent_policy-id_agent-num
    # print("Calling to policy mapping {}".format(agent_id))
    name, id, _ = agent_id.split("_")

    return "{}_{}".format(name, id)


def featurize(obs):
    raise NotImplemented
