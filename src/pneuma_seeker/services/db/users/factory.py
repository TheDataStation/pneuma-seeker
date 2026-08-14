# src/pneuma_seeker/services/db/users/factory.py — get_user_db() picks
# UserDB (config.AUTH_BACKEND == "local", default) or ExternalUserDB
# ("external") — see the migration plan. routers/auth.py, routers/chat.py,
# and routers/memory.py each construct their own module-level `user_db`
# from this instead of `UserDB(config, logger)` directly, which is the one
# change that makes AUTH_BACKEND=external cascade to every authenticated
# route across all three routers — see routers/auth.py's get_current_user
# (shared by chat.py/memory.py) plus chat.py's/memory.py's own additional
# direct group/permission lookups.
from pneuma_seeker.services.db.users.external_client import ExternalUserDB
from pneuma_seeker.services.db.users.manager import UserDB
from pneuma_seeker.shared.config import Config


def get_user_db(config: Config, logger) -> UserDB | ExternalUserDB:
    if config.AUTH_BACKEND == "external":
        return ExternalUserDB(config, logger)
    return UserDB(config, logger)
