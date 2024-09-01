import os
import time
import json
import random
import logging
import asyncio
import tiktoken
import networkx as nx
from typing import Dict, List, Union
from tqdm.asyncio import tqdm
from tqdm.auto import trange
from llm.localllm import LocalLLM
from graphragrec.embed.community.report import communityReport
from graphragrec.embed.community.summary import combineCommunityReports
from graphragrec.utils.usage import calculateUsages

logging.basicConfig(filename='movie_community_embedding_2.log',
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s')

encoding = tiktoken.encoding_for_model("gpt-4o")


def fetchCommunityData(G: nx.Graph, communities: Dict):
    community2data = {}
    for key, value in communities.items():
        if not value in community2data:
            community2data[value] = []
        if key in G:
            node_data = {
                neighbor: dict(G[key][neighbor])
                for neighbor in G[key]
            }
            community2data[value].append({
                "entity": key,
                "relations-claims": node_data
            })
        else:
            print(f"{key} not in graph")
    return community2data


def divideCommunity(community_content: List):
    print(f"Dividing community with {len(community_content)} items")
    index = 0
    community_batch = []
    while index < len(community_content):
        community_batch.append([])
        token_length = 0
        print(f">>>> INDEX: {index} | TOKEN LENGTH: {token_length}")
        while index < len(community_content):
            try:
                item_tokens = len(encoding.encode(f'{community_content[index]}'))
                print(f">>>>>>>> ITEM TOKENS: {item_tokens}")
                if token_length + item_tokens > 30000:
                    if token_length == 0:  # If the first item is already too large
                        print(f"WARNING: Item at index {index} is too large ({item_tokens} tokens). Adding it to its own batch.")
                        community_batch[-1].append(community_content[index])
                        index += 1
                    break
                token_length += item_tokens
                print(f">>>>>>>> INDEX: {index} | CUMULATIVE TOKEN LENGTH: {token_length}")
                community_batch[-1].append(community_content[index])
                index += 1
            except Exception as e:
                print(f"Error processing item at index {index}: {str(e)}")
                index += 1
    print(f"Community divided into {len(community_batch)} batches")
    return community_batch


def batchCommunities(community_data: Dict):
    print("Starting batchCommunities function")
    communitybatches = {}
    for cid, items in community_data.items():
        print(f"Processing community ID: {cid}")
        logging.info(f"> COMMUNITY ID: [{cid}]")
        community_batches = divideCommunity(items)
        print(f"Community {cid} divided into {len(community_batches)} batches")
        logging.info(
            f"> COMMUNITY ID: [{cid}] | TOTAL BATCHES: [{len(community_batches)}]"
        )
        communitybatches[cid] = community_batches
    print(f"Total communities processed: {len(communitybatches)}")
    return communitybatches


async def summarizeCommunity(llm: LocalLLM, model: str, community_id: int,
                             community_batches: List[List[Dict]]):
    if len(community_batches) == 1:
        cr, usage = await communityReport(llm, model, community_batches)
        if isinstance(cr, Dict):
            cr["community_id"] = community_id
        return cr, usage
    pbar = tqdm(total=len(community_batches),
                desc=f"Community Batch - {community_id}",
                colour="blue")
    community_reports = []
    usages = []
    batch_size = 2
    for ix in range(0, len(community_batches), batch_size):
        crs = await asyncio.gather(*[
            communityReport(llm, model, community_batches[c])
            for c in range(ix, ix + batch_size, 1)
            if c < len(community_batches)
        ])
        # logging.info(f"CRS: {crs}")
        u = calculateUsages([c[-1] for c in crs])
        usages.append(u)
        community_reports.extend([c[0] for c in crs])
        logging.info("COMMUNITY COOLDOWN>>>>>>")
        time.sleep(20)
        logging.info("COMMUNITY COOLDOWN DONE>>>>>>")
        pbar.update(batch_size)
    cr, usage = await combineCommunityReports(llm, model, community_reports)
    usages += [usage]
    usages = calculateUsages(usages)
    if isinstance(cr, Dict):
        cr["community_id"] = community_id
    return cr, usages


async def summarizeCommunities(llm: LocalLLM, model: str, communities: Dict):
    print("Starting summarizeCommunities function")
    communities_batched = batchCommunities(communities)
    print(f"Batched communities: {len(communities_batched)}")
    pbar = tqdm(total=len(communities_batched),
                desc="Summarizing Communities",
                colour="blue")
    usages = []
    comprehensive_communities_reports = {}
    community_ids = list(communities.keys())

    batch_size = 5
    flatten = lambda lst: [item for sublist in lst for item in sublist]

    async def execute_community_summary(community_id: int,
                                        community_batches: List[List[Dict]]):
        try:
            print(f"Processing community {community_id}")
            time.sleep(random.choice([0.2, 0.4, 0.5, 0.8, 0.9, 1.1]))
            cr, u = await summarizeCommunity(llm, model, community_id,
                                             community_batches)
            comprehensive_communities_reports[community_id] = {
                "data": flatten(community_batches),
                "report": cr
            }
            usages.append(u)
            pbar.update(1)
        except Exception as err:
            logging.exception(
                f"EXCEPTION: {str(err)}\nCOMMUNITY: {community_id}")
            pbar.update(1)

    async for ix in trange(0,
                           len(community_ids),
                           batch_size,
                           desc="Batch Summarization",
                           leave=False):
        try:
            print(f"Processing batch {ix}")
            await asyncio.gather(*[
                execute_community_summary(
                    community_ids[i], communities_batched[community_ids[i]])
                for i in range(ix, ix + batch_size, 1)
                if i < len(community_ids)
            ])
        except Exception as err:
            logging.exception(f"Exception: {str(err)}")
            pbar.update(1)
        logging.info("COOLDOWN>>>>")
        time.sleep(30)
        logging.info("COOLDOWN DONE>>>>")
    total_usage = calculateUsages(usages)
    return comprehensive_communities_reports, total_usage


if __name__ == "__main__":
    import pickle
    import json
    from configs import OPENAI_API_KEY
    
    print("Starting script execution")
    
    base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    graph_path = os.path.join(base_path, "output", "v9-gpt-4o-mini", "graph.gpickle")
    communities_path = os.path.join(base_path, "output", "v9-gpt-4o-mini", "communities.json")
    community_data_path = os.path.join(base_path, "output", "v9-gpt-4o-mini", "community-data.json")
    community_reports_path = os.path.join(base_path, "output", "v9-gpt-4o-mini", "community-reports.json")
    
    print(f"Loading graph from {graph_path}")
    with open(graph_path, "rb") as fp:
        G = pickle.load(fp)
    print(f"Loading communities from {communities_path}")
    with open(communities_path, "r") as fp:
        communities = json.load(fp)
    print("Fetching community data")
    community2data = fetchCommunityData(G, communities)
    print(f"TOTAL COMMUNITIES: {len(list(community2data.keys()))}")
    print(f"Saving community data to {community_data_path}")
    with open(community_data_path, "w") as fp:
        json.dump(community2data, fp, indent=4)
    print("Initializing LocalLLM")
    llm = LocalLLM(api_key=OPENAI_API_KEY)
    print("Starting summarizeCommunities")
    community_reports, total_usage = asyncio.run(
        summarizeCommunities(llm, "gpt-4o-mini", community2data))
    print(f"Saving community reports to {community_reports_path}")
    with open(community_reports_path, "w") as fp:
        json.dump(community_reports, fp, indent=4)
    print("Script execution completed")
