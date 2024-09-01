import os

base_path = os.path.dirname(os.path.abspath(__file__))

class EXTRACT:
    ENTITIES = open(os.path.join(base_path, "extract", "entity.txt")).read()
    RELATION = open(os.path.join(base_path, "extract", "relation.txt")).read()
    CLAIM = open(os.path.join(base_path, "extract", "claim.txt")).read()


class EMBED:
    COMMUNITY = open(os.path.join(base_path, "embed", "community.txt")).read()
    COMBINE = open(os.path.join(base_path, "embed", "communitycombine.txt")).read()


class QUERY:
    MAP = open(os.path.join(base_path, "query", "map.txt")).read()
    REDUCE = open(os.path.join(base_path, "query", "reduce.txt")).read()
    COMMUNICATE = open(os.path.join(base_path, "query", "communicate.txt")).read()
