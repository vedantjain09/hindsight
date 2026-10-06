from .calendar import CalendarEnv
from .files import FilesEnv
from .orders import OrdersEnv

ENVS = {"orders": OrdersEnv, "files": FilesEnv, "calendar": CalendarEnv}


def make_env(name: str, seed: int):
    return ENVS[name](seed)
