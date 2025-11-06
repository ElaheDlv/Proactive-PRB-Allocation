import inspect


class xAppBase:
    def __init__(self, ric=None):
        self.ric = ric
        self.enabled = True

    @property
    def xapp_id(self):
        return self.__class__.__name__

    @property
    def base_station_list(self):
        return self.ric.base_station_list

    @property
    def cell_list(self):
        return self.ric.cell_list

    @property
    def ue_list(self):
        return self.ric.ue_list

    def start(self):
        assert False, "Not implemented in base class"

    def step(self):
        # xApps can be implemented in an event-driven manner or in an simulation step-driven manner.
        # if this method is overridden, it will be called in each simulation step by the RIC.
        pass

    def to_json(self):
        # Get the source code of the actual class (including child classes)
        try:
            source_code = inspect.getsource(self.__class__)
        except Exception:
            source_code = None
        return {
            "xapp_id": self.xapp_id,
            "enabled": self.enabled,
            "source_code": source_code,
        }
