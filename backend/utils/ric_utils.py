class xAppControlAction:

    ACTION_TYPE_HANDOVER = "handover"

    def __init__(self, action_type, action_data):
        self.action_type = action_type
        self.action_data = action_data
