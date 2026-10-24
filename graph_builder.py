
import math

import numpy as np
from langchain_core.prompts import PromptTemplate
import ast
import re
import hashlib, os, json


GRAPH_BUILD_CACHE_VERSION = "v2"


def _stage_path(cache_dir, props_hash, name):
    return os.path.join(cache_dir, f".build_{GRAPH_BUILD_CACHE_VERSION}_{props_hash}_{name}.json")


def _load_stage(cache_dir, props_hash, name):
    path = _stage_path(cache_dir, props_hash, name)
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError, TypeError):
        return None


def _save_stage(cache_dir, props_hash, name, value):
    os.makedirs(cache_dir, exist_ok=True)
    path = _stage_path(cache_dir, props_hash, name)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False)
    os.replace(tmp_path, path)


def _stage_log(stage, message):
    print(f"[{stage}/9] {message}", flush=True)





def pre_fun_count(Propositions, model) -> dict:

    """
    Count entity occurrences together with their proposition ids. Entities can be
    grouped together only when their propositions share extend_key_list entries
    (semantic disambiguation). In effect, identical 'entity_name' values with
    intersecting 'extend_key_list' values form one entity cluster.
    """
    entity_dic = {}
    # add_idx = []
    for item in Propositions:
        _id = item['idx']
        # if _id not in add_idx:
            # add_idx.append(_id)
        current_tags = set(item['extend_key_list'])
        for entity in item['entity_list']:
            entity = entity.strip().replace("'", "").replace('"', "")
            if entity not in entity_dic:
                # First insertion into an empty dict, or first time this entity name appears.
                entity_dic.update({entity: [
                    {
                        'tags': current_tags.copy(),
                        'idx': [_id],
                        'count': 1
                    }
                ]})
                continue

            is_merged = False
            for i, ever_tag in enumerate(entity_dic[entity]):
                # if (semantic_set_similarity(current_tags, ever_tag["tags"], model) > 0.8
                #                                or
                #                                ever_tag["tags"]&current_tags):
                if (semantic_set_similarity(current_tags, ever_tag["tags"], model) > 0.8):
                    entity_dic[entity][i]["tags"] |= current_tags
                    entity_dic[entity][i]["idx"].append(_id)
                    entity_dic[entity][i]["count"] += 1
                    is_merged = True
                    break
            if not is_merged:
                group = {
                    'tags': current_tags,
                    'idx': [_id],
                    'count': 1
                }
                entity_dic[entity].append(group)
    # print(entity_dic)
    return entity_dic

def func_count(pre_entity_dic) -> dict:
    """
    Rebuild the dict so that split entity names get their own dict entry with
    encoding, while guaranteeing each proposition node appears exactly once and
    is kept only under the highest-frequency entity name.
    Logic:
    1. Count the "unique proposition IDs" and "total frequency (unique propositions)" per entity.
    2. Sort entities by frequency in descending order.
    3. Walk the sorted entities and assign each entity the proposition IDs not
       already taken by higher-frequency entities.
    4. Recompute the final proposition counts and frequencies.
    """

    tags_list = []
    entity_name_list = []
    entity_name_idx = []
    entity_name_count_list = []

    for entity_name, entity_clusters in pre_entity_dic.items():
        current_tags = set()
        for cluster in entity_clusters:
            tags = set(cluster['tags'])
            current_tags.update(tags)
        tags_list.append(list(current_tags))

    for entity_name, entity_clusters in pre_entity_dic.items():
        entity_name_list.append(entity_name)
        total_count = 0
        all_idx = set()
        for cluster in entity_clusters:
            all_idx.update(cluster["idx"])
            total_count += cluster["count"]

        entity_name_idx.append(list(all_idx))
        entity_name_count_list.append(total_count)

    pairs = zip(entity_name_list, entity_name_idx, entity_name_count_list)
    sorted_list = sorted(pairs, key=lambda x: x[2], reverse=True)

    sorted_name, sorted_idx, sorted_count = zip(*sorted_list)

    entity_name_list = list(sorted_name)
    entity_name_idx = list(sorted_idx)
    entity_name_count_list = list(sorted_count)

    new_entity_name_idx = []
    new_entity_name_count_list = []
    seed_list = []

    for idx_group in entity_name_idx:
        new_group = []
        for i in idx_group:
            if i not in seed_list:
                seed_list.append(i)
                new_group.append(i)
        new_entity_name_idx.append(new_group)
        new_entity_name_count_list.append(len(new_group))


    pairs = zip(entity_name_list, new_entity_name_idx, new_entity_name_count_list)
    sorted_list = sorted(pairs, key=lambda x: x[2], reverse=True)

    sorted_name, sorted_idx, sorted_count = zip(*sorted_list)

    entity_name_list = list(sorted_name)
    entity_name_idx = list(sorted_idx)
    entity_name_count_list = list(sorted_count)

    final_entity = {}
    for i, name in enumerate(entity_name_list):
        if entity_name_count_list[i] > 0:
            final_entity[name] = {
                "tags": tags_list[i],
                "idx": entity_name_idx[i],
                "count": entity_name_count_list[i]
            }
        else:
            break

        # final_entity.append(new_entity_group)
    # print(final_entity)
    return final_entity

def make_sureN(entity_dict):
    """N is the frequency threshold: we cluster the core-entity keywords whose
    occurrence count reaches the top-80% quantile (highest frequency)."""

    count = len(entity_dict)
    N_80_idx = math.ceil(count * 0.8)

    entity_list = list(entity_dict.items())  # [(entity_name, {'idx': [...], 'count': n}), ...]

    if N_80_idx >= len(entity_list):
        N_80_idx = len(entity_list) - 1

    N_80_key = list(entity_dict.keys())[N_80_idx]
    N_80_count = entity_dict[N_80_key]["count"]

    if N_80_count == 1:
        N_80_count = 2
    idx = 0

    for index in range(len(entity_list) - 1, -1, -1):  # walk backwards
        current_count = list(entity_list[index])[1]['count']
        if current_count >= N_80_count:
            idx = index
            break
    # print(N_80_count, idx)

    return N_80_count, idx

def build_e_balls(entity_dict, embeddings, N, entity_idx, Propositions):
    """
    Core granular-ball clustering function (modified).
    Key change: the radius is computed from the original high-dimensional
    embeddings instead of 2-D vectors after dimensionality reduction.
    high_dim_embeddings: original high-dimensional embedding vectors (not reduced).
    """

    E_ball = []

    count = 0
    prop_idx_map = {item['idx']: i for i, item in enumerate(Propositions)}  # proposition idx -> embedding index

    for i, (entity_name, item) in enumerate(entity_dict.items()):
        if i == 0 or (i + 1) % 500 == 0:
            print(f"[4/8] entity cluster {i + 1}/{len(entity_dict)}", flush=True)
        if item["count"] >= N:  # only process entity clusters occurring >= N times
            current_indices = item["idx"]  # proposition idx list of this entity cluster

            center_idx, radius = get_cluster_center_and_radius(current_indices, embeddings)

            entity_name, tags = k_tags(current_indices, Propositions)
            # store the ball info (radius computed in high-dimensional space)
            ball = {
                "idx": f"C_{count}",
                "center_idx": center_idx,
                "all_indices": current_indices,
                "radius": radius,  # radius computed in high-dimensional space
                "entity_list": entity_name,
                "tags": tags
            }
            count += 1
            E_ball.append(ball)

    return E_ball

def idx_str_2_idx_int(a_idx_str):
    a_idx_int = []
    for str_idx in a_idx_str:
        int_idx = int(str_idx.split("_")[-1])
        a_idx_int.append(int_idx)
    return a_idx_int

def get_idx_entity(idx_int, Propositions):
    entity_list = []
    for idx in idx_int:
        entity = set(Propositions[idx]["entity_list"])
        entity = list(entity)
        entity_list.extend(entity)

    return entity_list

def split_e_into_d_balls(E_balls, Propositions, embeddings, model):

    D_balls = []

    count = -1
    for idx, ball in enumerate(E_balls):
        if idx == 0 or (idx + 1) % 100 == 0:
            print(f"[6/8] E ball {idx + 1}/{len(E_balls)}", flush=True)
        A_idx_list_used = set()
        # entity_list = ball["entity_list"]
        E_2_D_list = []

        current_all_indices = ball["all_indices"]

        for i in range(len(ball["all_indices"])):
            i_str = ball["all_indices"][i]
            if i_str in A_idx_list_used:
                continue
            all_str_idx = [i_str]
            A_idx_list_used.add(i_str)

            i_int = int(i_str.split("_")[-1])
            i_tag = Propositions[i_int]["extend_key_list"]
            i_prop = Propositions[i_int]["proposition"]
            comment_tags = i_tag.copy()

            for j in range(i + 1, len(ball["all_indices"])):
                j_str = ball["all_indices"][j]
                if j_str in A_idx_list_used:
                    continue
                j_int = int(j_str.split("_")[-1])
                j_tag = Propositions[j_int]["extend_key_list"]
                j_prop = Propositions[j_int]["proposition"]

                score = semantic_set_similarity(i_tag, j_tag, model)

                vec_i = embeddings[i_int]
                vec_j = embeddings[j_int]
                prop_sim = np.dot(vec_i, vec_j) / (np.linalg.norm(vec_i) * np.linalg.norm(vec_j))

                if prop_sim > 0.75 or (score > 0.8):
                    all_str_idx.append(j_str)
                    comment_tags = list(set(comment_tags + j_tag))
                    A_idx_list_used.add(j_str)
            count += 1
            if len(all_str_idx) > 1:
                center_idx, radius = get_cluster_center_and_radius(all_str_idx, embeddings)
            else:
                center_idx = all_str_idx[0]
                radius = 0

            b_idx_int = idx_str_2_idx_int(all_str_idx)
            b_entity_list = list(dict.fromkeys(get_idx_entity(b_idx_int, Propositions)))

            time_indices_in_ball = []
            for idx_str in all_str_idx:
                idx_int = int(idx_str.split("_")[-1])
                if Propositions[idx_int].get("time_entity_list"):
                    time_indices_in_ball.append(idx_str)

            new_ball = {
                "idx": f"B_{count}",
                "center_idx": center_idx,
                "all_indices": all_str_idx,
                "radius": radius,
                "entity_list": b_entity_list,
                "tags": comment_tags,
                "time_indices": time_indices_in_ball,
            }
            E_2_D_list.append(f"B_{count}")
            D_balls.append(new_ball)
        E_balls[idx].update({'d_indices': E_2_D_list})
    return D_balls, E_balls

def cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))

def print_final_ball_new(final_ball, Propositions, fen):
    List = []

    for ball in final_ball:
        ball_group_list = []
        for idx in ball:
            prop_idx = int(idx.split('_')[-1])
            prop = Propositions[prop_idx]['proposition']
            entity_list = Propositions[prop_idx]['entity_list']
            extend_key_list = Propositions[prop_idx]['extend_key_list']
            group = {
                'idx': idx,
                'proposition': prop,
                'entity_list': entity_list,
                'extend_key_list': extend_key_list
            }
            ball_group_list.append(group)
        List.append(ball_group_list)

    return List

def cluster_one_node_new(none_clustering_list, fen, Propositions, threshold=0.6):
    used_idx = []
    final_balls = []
    one_nodes = []

    for i, ball_a in enumerate(none_clustering_list):
        if ball_a['idx'][0] in used_idx: continue

        # Find other isolated points semantically very close to ball_a.
        new_ball = [ball_a['idx'][0]]
        for j in range(i + 1, len(none_clustering_list)):
            ball_b = none_clustering_list[j]
            if ball_b['idx'][0] in used_idx: continue

            dist = cosine_distance(ball_a['embedding'], ball_b['embedding'])
            if dist < threshold:  # very close: a missed topic cluster
                new_ball.append(ball_b['idx'][0])
                if ball_a['idx'][0] in used_idx:
                    used_idx.append(ball_b['idx'][0])
                else:
                    used_idx.append(ball_a['idx'][0])
                    used_idx.append(ball_b['idx'][0])
        if len(new_ball) == 1:
            one_nodes.append(ball_a['idx'][0])
        else:
            final_balls.append(new_ball)
        # final_balls = [[cluster set of the first group of unclustered proposition nodes],
        #                [cluster set of the second group of unclustered proposition nodes],
        #                 ]

    list = print_final_ball_new(final_balls, Propositions, fen)


    return list, one_nodes

def patch_unclustered_nodes_new(E_balls, None_Clustering, embeddings, Propositions):
    balls_list = []

    for idx_int in None_Clustering:
        # Parse the proposition id (e.g. 'A_0') into an index matching Propositions.
        embedding = embeddings[idx_int]
        current_ball = {
            'idx': [f"A_{idx_int}"],  # keep the original proposition id (e.g. 'A_0') for later mapping
            'embedding': embedding
        }
        balls_list.append(current_ball)

    one_ball_list, one_nodes = cluster_one_node_new(balls_list, len(E_balls), Propositions)

    return one_ball_list, one_nodes

def func_res(res: str):
    re1 = "entity_list\s*=\s*(.*?)(?=\s*,?\s*extend_key_list)"
    entity_list_str = re.findall(re1, res, re.DOTALL)
    entity_list = ast.literal_eval(entity_list_str[0])
    # print(entity_list)

    re2 = "extend_key_list\s*=\s*(.*?)(?=\n|$)"
    extend_key_list_str = re.findall(re2, res, re.DOTALL)
    extend_key_list = ast.literal_eval(extend_key_list_str[0])

    return entity_list, extend_key_list

def build_e_balls_second_pass(one_B_balls_cluster, entity_dict, exting_Balls, embeddings, model, llm):
    prompt_str = """
    You are a professional expert in knowledge graph entity alignment. Your task is to generate a list of 1 to 3 representative entities along with 1 to 3 domain labels for a group of semantically similar propositions.

    Background:

    These propositions are clustered together because they discuss the same specific topic, event, or organization, even if they do not contain identical entity names.

    Input Propositions:
    {Propositions}

    Extraction Requirements:

    1.Prioritize identifying explicit entities: Extract the most important proper nouns present in the text, such as names of organizations, people, specific locations, or specific time periods.

    2.Resolve implicit topics: If the proposition text uses general terms (e.g., "league", "team", "fire", "magazine"), infer their specific official names based on the context of other entities. For example, if you see "Kyiv Generals", the "league" most likely refers to the "Ukrainian Hockey Championship".

    3.Prioritize specificity:
    -Poor entity names: "Sports", "Education", "Disaster" (overly broad/upper-level concepts).
    -Good entity names: "FSHF", "Wesley College", "Highway Complex Fire".

    4.Adopt a unification strategy: Since these propositions are complementary descriptions, list the 1 to 3 distinct entities that best represent the content of the entire set of propositions.

    5.Extract domain labels: For the content of this cluster, extract 1 to 3 upper-level conceptual terms as domain labels. Examples: "Sports", "Higher Education", "Natural Disasters", "Albanian History".

    Output Format:
    Return only a Python string list.
    Example: entity_list=["Entity Name 1","Entity Name 2","Entity Name 3"],extend_key_list=["Domain Label 1", "Domain Label 2","Domain Label 3"]
    """
    prompt = PromptTemplate(
        input_variables=["Propositions"],
        template=prompt_str
    )

    chain = prompt | llm

    temp_new_balls = []

    last_idx = len(exting_Balls) - 1

    for i, group in enumerate(one_B_balls_cluster):
        indices_list = []
        Propositions_list = []

        for item in group:
            idx = item["idx"]
            # idx = item["idx"].split("_")[-1]
            indices_list.append(idx)
            Propositions_list.append(item['proposition'])

        res = chain.invoke({"Propositions": Propositions_list})
        res_str = str(res.content.replace('```', "'"))
        group_entity_list, group_key_list = func_res(res_str)

        ball_date = {
            'all_indices': indices_list,
            "tags": group_key_list,
            "entity_list": group_entity_list
        }
        temp_new_balls.append(ball_date)

    final_new_balls = []
    for new_ball in temp_new_balls:
        merged = False
        new_entity_set = set(new_ball['entity_list'])

        for existing_ball in exting_Balls:
            existing_entity_set = set(existing_ball['entity_list'])
            if new_entity_set & existing_entity_set or semantic_set_similarity(new_entity_set, existing_entity_set,
                                                                               model) > 0.7:
                # Similar to an existing ball: merge into it.
                existing_ball['all_indices'].extend(new_ball['all_indices'])
                existing_ball['tags'] = list(set(existing_ball['tags'] + new_ball['tags']))
                center_idx, radius = get_cluster_center_and_radius(existing_ball['all_indices'], embeddings)
                existing_ball['center_idx'] = center_idx
                existing_ball['radius'] = radius

                merged = True
                break  # merged into one ball; stop searching

        # Merged into no existing ball: becomes a formal new E ball.
        if not merged:
            last_idx += 1
            center_idx, radius = get_cluster_center_and_radius(new_ball['all_indices'], embeddings)
            final_ball = {
                "idx": f"C_{last_idx}",
                "center_idx": center_idx,
                'all_indices': new_ball['all_indices'],
                "radius": radius,
                "tags": new_ball['tags'],
                "entity_list": new_ball['entity_list']
            }
            final_new_balls.append(final_ball)
    # print('clusters of entity/domain granular balls extracted from mixed propositions:')
    # print(one_balls)

    return final_new_balls

def unclustered_nodes_func(entity_dict, E_balls, embeddings, fen, model, llm, Propositions):

    clustered_indices = set()
    one_node_embeddings = []
    center_radius_list = []
    center_embeddings = []
    # track the E-ball indices that changed
    C_idx = set()

    # First collect the ids of unclustered propositions.
    for item in E_balls:
        for str_idx in item['all_indices']:
            int_idx = int(str_idx.split('_')[-1])
            clustered_indices.add(int_idx)
    one_node_list = [i for i in range(fen) if i not in clustered_indices]

    # fetch the vectors of the corresponding ids
    for idx in one_node_list:
        one_node_embeddings.append(embeddings[idx])
    one_node_embeddings = np.array(one_node_embeddings)

    # fetch the radius (angle) list and center-vector list of every E ball
    for item in E_balls:
        center_idx = int(item["center_idx"].split("_")[-1])
        radius = item["radius"]
        center_embedding = embeddings[center_idx]
        center_radius_list.append(radius)
        center_embeddings.append(center_embedding)
    center_embeddings = np.array(center_embeddings)

    # compute the angles between unclustered proposition nodes and all E-ball
    # centers; find the ball whose radius covers the node (smallest angle among
    # those within radius).
    one_node_embs_length = np.linalg.norm(one_node_embeddings, axis=1, keepdims=True)
    one_node_embs_length = np.where(one_node_embs_length == 0, 1e-8, one_node_embs_length)
    center_embs_length = np.linalg.norm(center_embeddings, axis=1, keepdims=True)
    center_embs_length = np.where(center_embs_length == 0, 1e-8, center_embs_length)


    # compute the cosine-similarity matrix
    cos_matrix = np.dot(one_node_embeddings, center_embeddings.T) / (one_node_embs_length * center_embs_length.T)
    cos_matrix = np.clip(cos_matrix, -1.0, 1.0)

    # convert cosine to 1 - cosine, which plays the role of "radius" (angular distance)
    radius_cos = 1 - np.array(cos_matrix)

    one_node_num = len(one_node_list)
    new_one_onde_list = []

    for i in range(one_node_num):
        if i == 0 or (i + 1) % 5000 == 0:
            print(f"[5/8] unclustered proposition {i + 1}/{one_node_num}", flush=True)
        node_cos = radius_cos[i].tolist()
        ball_indices = list(range(len(node_cos)))

        matched = False
        while (len(node_cos) > 0 and not matched):

            min_cos_val = min(node_cos)
            min_cos_idx = node_cos.index(min_cos_val)
            C_balls_idx = ball_indices[min_cos_idx]

            if E_balls[C_balls_idx]["radius"] > min_cos_val:
                E_balls[C_balls_idx]["all_indices"].append(f"A_{one_node_list[i]}")
                matched = True
                C_idx.add(C_balls_idx)

            else:
                node_cos.pop(min_cos_idx)
                ball_indices.pop(min_cos_idx)
        if not matched:
            new_one_onde_list.append(one_node_list[i])

    # eliminate the remaining isolated points:
    final_unclustered_list = []

    for unclustered_idx in new_one_onde_list:
        node_embedding = embeddings[unclustered_idx]
        node_tags = set(Propositions[unclustered_idx]["extend_key_list"])

        # compute the distance from this isolated node to every E-ball center and
        # find the nearest ball (even when the distance already exceeds its radius).
        # Then compare the node's extend_key_list with that ball's tags: if they
        # intersect (or semantic_set_similarity is high), absorb the node.
        distances = []
        for c_idx, center_emb in enumerate(center_embeddings):
            cos = np.dot(node_embedding, center_emb) / (np.linalg.norm(node_embedding) * np.linalg.norm(center_emb))
            distances.append(1 - cos)

        close_c_ball_idx = np.argmin(distances)
        close_c_ball = E_balls[close_c_ball_idx]
        # print(close_c_ball["tags"])  # e.g. ['Radio Broadcasting', 'Media Industry', 'Music Industry', 'Corporate Leadership']

        ball_tags = set()
        for tag in close_c_ball["tags"]:
            if isinstance(tag, list):
                ball_tags.update(tag)
            else:
                ball_tags.add(tag)

        if node_tags & ball_tags or semantic_set_similarity(node_tags, ball_tags, model) > 0.65:
            E_balls[close_c_ball_idx]["all_indices"].append(f"A_{unclustered_idx}")
            new_tags_set = set(close_c_ball["tags"]) | node_tags
            E_balls[close_c_ball_idx]["tags"] = list(new_tags_set)

            C_idx.add(close_c_ball_idx)

        else:
            final_unclustered_list.append(unclustered_idx)

    new_one_onde_list = final_unclustered_list

    # refresh the modified E-ball info
    for j in C_idx:
        current_indices = E_balls[j]["all_indices"]
        center_idx, radius = get_cluster_center_and_radius(current_indices, embeddings)
        E_balls[j]["center_idx"] = center_idx
        E_balls[j]["radius"] = radius

    C_second_balls, one_node_new = patch_unclustered_nodes_new(E_balls, new_one_onde_list, embeddings, Propositions)

    # re-extract entities and re-assign domains for the remaining unclustered nodes
    C_balls_second = build_e_balls_second_pass(C_second_balls, entity_dict, E_balls, embeddings, model, llm)

    E_balls = E_balls + C_balls_second

    return E_balls, one_node_new



def link_e_balls_by_shared_entities(global_candidates, C_entity_map, Propositions):
    """
    :param global_candidates: boundary nodes of all E balls, each with its entity list
    :param Propositions: original proposition list
    :return: edge list info
    """
    edges = []
    edges_key = set()
    entity_to_c_balls = {}
    for c_idx, entities in C_entity_map.items():
        for entity in entities:
            entity_to_c_balls.setdefault(entity, set()).add(c_idx)

    # track the proposition node ids each ball has already contributed
    contribute_E_ball = {}

    for item in global_candidates:
        c1 = item["C_ball_idx"]
        # ---- disabled Plan A: one boundary node per ball ----
        # if c1 in contribute_E_ball:
        #     continue
        # ----------------------------------------------------
        max_idx_int = int(item["max_idx"].split("_")[-1])
        max_idx_str = item["max_idx"]
        max_idx_entity_list = item["entity_list"]

        entity_set = set(entity.lower().strip() for entity in max_idx_entity_list)
        is_edge_formed = False

        candidate_c2 = set()
        for entity in entity_set:
            candidate_c2.update(entity_to_c_balls.get(entity, ()))
        for c2 in candidate_c2:
            value = C_entity_map[c2]
            edge_key = tuple(sorted([c1, c2]))
            if c1 == c2 or edge_key in edges_key:
                continue
            if entity_set & value:
                edge = {
                    "node_E_1": c1,
                    "node_E_2": c2,
                    "relation": Propositions[max_idx_int]["proposition"],
                    "shared_entities": list(entity_set & value),
                    "prop_idx": max_idx_str,
                    "type": "global_boundary_bridge"
                }
                edges.append(edge)
                edges_key.add(edge_key)
                is_edge_formed = True

    # ---------------------- Plan A ----------------------
    #     if is_edge_formed:
    #         contribute_E_ball[c1] = max_idx_str
        # ---------------------- Plan B ----------------------
        if is_edge_formed:
            # append the contributed node to the list
            if c1 not in contribute_E_ball:
                contribute_E_ball[c1] = []
            if max_idx_str not in contribute_E_ball[c1]:
                contribute_E_ball[c1].append(max_idx_str)



    return edges, edges_key, contribute_E_ball

def cal_cos(emb1, emb2):
    emb1 = np.array(emb1)
    emb2 = np.array(emb2)

    norm1 = np.linalg.norm(emb1)
    norm2 = np.linalg.norm(emb2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    cos = np.dot(emb1, emb2) / (norm1 * norm2)
    return cos

def make_boundary_edges(E_balls, D_balls, Propositions, embeddings):
    """
    Global outliers: rank all E-ball nodes together and build edges starting
    from the most peripheral ones.
    :param max_edges_per_ball: how many propositions a E ball may contribute as
                               edges (default 1, to avoid draining a ball)
    """
    # 1. map each E ball to its entity list
    C_entity_map = {
        ball["idx"]: set(str(e).lower().strip() for e in ball["entity_list"]) for ball in E_balls
    }
    entity_to_c_balls = {}
    for c_idx, entities in C_entity_map.items():
        for entity in entities:
            entity_to_c_balls.setdefault(entity, set()).add(c_idx)

    # 2. find, per E ball, the propositions farthest from the center vector
    # (with their entity lists); collect all non-center nodes of all E balls
    # into the global candidate pool
    global_candidates = []
    for ball in E_balls:
        center_idx_str = ball["center_idx"]
        center_emb = embeddings[int(center_idx_str.split("_")[-1])]

        for idx_str in ball["all_indices"]:
            if idx_str == center_idx_str:
                continue
            idx_int = int(idx_str.split("_")[-1])
            emb = embeddings[idx_int]
            cos_sim = cal_cos(emb, center_emb)
            dist = 1.0 - cos_sim  # 1 - cos, i.e. the angular distance from the center


            entity_list = Propositions[idx_int]["entity_list"]
            entity_set = set(e.lower().strip() for e in entity_list)

            # [Change 1]: count how many *other* E balls each entity of this node
            # can intersect with.
            cross_ball_ids = set()
            for entity in entity_set:
                cross_ball_ids.update(entity_to_c_balls.get(entity, ()))
            cross_ball_ids.discard(ball["idx"])
            cross_ball_count = len(cross_ball_ids)

            # relaxed threshold: any node not glued to the center may try to
            # form an edge.
            if dist < 0.3 and cross_ball_count == 0:
                continue

            group = {
                "C_ball_idx": ball["idx"],
                "max_idx": idx_str,
                "cos_max": dist,  # 1 - cos; larger means a wider angle from the center
                "entity_list": entity_list,
                "cross_count": cross_ball_count  # number of balls sharing entities with this node
            }
            global_candidates.append(group)

    # 3. sort boundary nodes by their angular distance from the center in
    # descending order; wide-angle nodes form edges first.
    global_candidates.sort(key=lambda x: (x["cross_count"], len(x["entity_list"]), x["cos_max"]), reverse=True)
    # print(global_candidates)

    # 4. check for shared entities with other balls; build an (undirected) edge
    # and record the already-linked undirected pairs.
    edges, edges_key, contribute_E_ball = link_e_balls_by_shared_entities(global_candidates, C_entity_map, Propositions)
    # Keep nohup logs manageable when the graph contains many edges.
    print(f"edges count: {len(edges)}")
    print(f"edge samples: {edges[:5]}")

    # 5. update E_balls and D_balls
    E_balls, D_balls = update_balls_after_edge_extraction(E_balls, D_balls, contribute_E_ball, embeddings, Propositions)

    return E_balls, D_balls, edges, edges_key

def update_balls_after_edge_extraction(E_balls, D_balls, contribute_E_ball, embeddings, Propositions):
    """
    Remove the proposition nodes extracted as edges from the original C and B
    balls and refresh their attributes.
    :return: updated E_balls, D_balls
    """

    d_balls_dict = {d['idx']: d for d in D_balls}

    for ball in E_balls:
        c_idx = ball['idx']

        # If this E ball contributed boundary nodes (was modified).
        if c_idx in contribute_E_ball:
            # removed_node_str = contribute_E_ball[c_idx]  # Plan A
            removed_nodes_list = contribute_E_ball[c_idx]  # Plan B

            # --------------- Plan A1 (aligned with Plan A of
            # link_e_balls_by_shared_entities): remove the node from the E ball.
            # if removed_node_str in ball['all_indices']:
            #     ball['all_indices'].remove(removed_node_str)

            # Plan B (aligned with Plan B of link_e_balls_by_shared_entities)
            for removed_node_str in removed_nodes_list:
                if removed_node_str in ball['all_indices']:
                    ball['all_indices'].remove(removed_node_str)

            # 2. refresh the E ball's center, radius, entities, and tags
            if len(ball['all_indices']) > 0:

                new_center, new_radius = get_cluster_center_and_radius(ball['all_indices'], embeddings)
                ball['center_idx'] = new_center
                ball['radius'] = new_radius

                new_entities, new_tags = k_tags(ball['all_indices'], Propositions)
                ball['entity_list'] = new_entities
                ball['tags'] = new_tags
            else:
                # extreme-case fallback: the ball was drained empty
                ball['radius'] = 0.0
                ball['entity_list'] = []
                ball['tags'] = []

            # 3. remove the node from D balls and drop emptied D balls
            if 'd_indices' in ball:
                empty_b_balls = []
                for d_idx_str in ball['d_indices']:
                    if d_idx_str in d_balls_dict:
                        d_ball = d_balls_dict[d_idx_str]
                        # delete the node from this D ball

                        #---------------- Plan A ----------------
                        # if removed_node_str in d_ball['all_indices']:
                        #     d_ball['all_indices'].remove(removed_node_str)
                        #
                        #     # if the D ball becomes empty, mark it for deletion
                        #     if len(d_ball['all_indices']) == 0:
                        #         empty_b_balls.append(d_idx_str)

                        #---------------- Plan B ----------------
                        for removed_node_str in removed_nodes_list:
                            if removed_node_str in d_ball['all_indices']:
                                d_ball['all_indices'].remove(removed_node_str)

                            # if the D ball becomes empty, mark it for deletion
                        if len(d_ball['all_indices']) == 0:
                            if d_idx_str not in empty_b_balls:
                                empty_b_balls.append(d_idx_str)

                # remove emptied D balls from the E ball's member list and the global dict
                for empty_b in empty_b_balls:
                    ball['d_indices'].remove(empty_b)
                    del d_balls_dict[empty_b]

                    # convert the refreshed, complete D_balls dict back to a list
    updated_B_balls = list(d_balls_dict.values())

    return E_balls, updated_B_balls

def summarize_e_balls(E_balls, llm, Propositions):

    for ball in E_balls:
        all_indices = ball["all_indices"]
        summary = map_reduce_summary(all_indices, Propositions, llm, 50)
        ball["summary"] = summary

    return E_balls

def get_cluster_center_and_radius(idx_list, embeddings):
    embedding_list = []

    for i in idx_list:
        idx = int(i.split('_')[-1])
        embedding_list.append(embeddings[idx])
    embedding_list = np.array(embedding_list)

    embeddings_mean = np.mean(embedding_list, axis=0)
    # --- cosine angle of each point to the mean vector ---
    # 1. norm of each vector
    length_embedding = np.linalg.norm(embedding_list, axis=1)
    length_embedding = np.where(length_embedding == 0, 1e-8, length_embedding)

    # mean vector

    mean_length_embedding = np.linalg.norm(embeddings_mean)
    if mean_length_embedding != 0:
        mean_length_embedding = mean_length_embedding
    else:
        mean_length_embedding = 1e-8

    # cosine of each vector to the mean vector
    cos_to_mean = np.dot(embedding_list, embeddings_mean) / (length_embedding * mean_length_embedding)

    # pick the vector with the smallest angle to the mean as the center
    center_index = np.argmax(cos_to_mean)
    center_vector = embedding_list[center_index]
    center_idx = idx_list[center_index]

    # radius is the angle to the center vector
    center_length = length_embedding[center_index]
    cos_list = np.dot(embedding_list, center_vector) / (center_length * length_embedding)

    cos_sims = np.clip(cos_list, -1.0, 1.0)
    min_cos = float(np.min(cos_list))  # smallest cos -> widest angle from the center

    radius = 1.0 - min_cos

    min_radius = 0.01
    if radius < min_radius:
        radius = min_radius

    return center_idx, radius


def semantic_set_similarity(set1, set2, model):
    """Merge tag sets that are semantically similar."""

    # vec1 = np.mean(model.encode(list(set1)), axis=0)
    # vec2 = np.mean(model.encode(list(set2)), axis=0)
    #
    # sim = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    # return sim
    if not set1 or not set2:
        return 0.0

        # ensure set elements are strings and convert to lists (avoids
        # set-related type issues)
    list1 = [str(item) for item in set1 if item]  # filter out empty elements
    list2 = [str(item) for item in set2 if item]

    # re-check non-emptiness (guard against all-empty-string sets)
    if not list1 or not list2:
        return 0.0

    # Cache tag embeddings: this function is called repeatedly during clustering.
    cache = getattr(model, "_gbrag_tag_embedding_cache", None)
    if cache is None:
        cache = {}
        try:
            setattr(model, "_gbrag_tag_embedding_cache", cache)
        except Exception:
            pass
    key1 = tuple(sorted(list1))
    key2 = tuple(sorted(list2))
    if key1 not in cache:
        cache[key1] = np.mean(model.encode(list1), axis=0)
    if key2 not in cache:
        cache[key2] = np.mean(model.encode(list2), axis=0)
    vec1 = cache[key1]
    vec2 = cache[key2]

    # cosine similarity with divide-by-zero protection
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0:
        return 0.0

    sim = np.dot(vec1, vec2) / (norm1 * norm2)
    return sim

def k_tags(current_indices, Propositions):
    idx_list = []
    for str_id in current_indices:
        idx = int(str_id.split("_")[-1])
        idx_list.append(idx)


    all_tags = []
    tag_count = []
    for i in idx_list:
        current_tag_list = Propositions[i]["extend_key_list"]
        for tag in current_tag_list:
            if tag not in all_tags:
                all_tags.append(tag)
                tag_count.append(1)
            else:
                tag_id = all_tags.index(tag)
                tag_count[tag_id] += 1
    peer = zip(all_tags, tag_count)
    sorted_peer = sorted(peer, key=lambda x: x[1], reverse=True)
    final_tag_list = [item[0] for item in sorted_peer][:4]

    all_entity = []
    entity_count = []
    for i in idx_list:
        current_entity_list = Propositions[i]["entity_list"]
        for entity in current_entity_list:
            if entity not in all_entity:
                all_entity.append(entity)
                entity_count.append(1)
            else:
                entity_id = all_entity.index(entity)
                entity_count[entity_id] += 1
    peer_0 = zip(all_entity, entity_count)
    sorted_peer_0 = sorted(peer_0, key=lambda x: x[1], reverse=True)
    # final_entity_list = [item[0] for item in sorted_peer_0][:3]  # keep the top-3 high-frequency entities
    final_entity_list = [item[0] for item in sorted_peer_0]  # keep all

    return final_entity_list, final_tag_list

# ==========================================
# Vector-database bridge (supports E, D, A multi-level retrieval)
# ==========================================
class VectorDBHelper:
    def __init__(self, persist_directory="./rag_chroma_db"):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_directory)

        # shared HNSW configuration
        hnsw_metadata = {
            "hnsw:space": "cosine",
            "hnsw:M": 16,
            "hnsw:construction_ef": 100
        }

        # two independent retrieval collections, one per level
        self.c_collection = self.client.get_or_create_collection(name="e_balls_collection", metadata=hnsw_metadata)
        self.a_collection = self.client.get_or_create_collection(name="a_props_collection", metadata=hnsw_metadata)


    def add_e_balls(self, e_balls, embedding_model):
        if not e_balls: return
        ids, docs, embs, metas = [], [], [], []
        for ball in e_balls:
            ids.append(ball["idx"])
            doc_text = ball.get("summary", f"Entities: {', '.join(ball['entity_list'])}. Tags: {', '.join(ball['tags'])}.")
            docs.append(doc_text)
            embs.append(embedding_model.encode(doc_text).tolist())
            metas.append({"tags": ", ".join(ball["tags"]), "entity_list": ", ".join(ball["entity_list"])})

        batch_size = self.client.get_max_batch_size()
        total_batches = (len(ids) + batch_size - 1) // batch_size
        print(f"Writing {len(ids)} E-balls in {total_batches} batch(es), max {batch_size} per batch...")
        for i in range(0, len(ids), batch_size):
            self.c_collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embs[i:i + batch_size],
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
            )
        print(f"[OK] Persisted {len(e_balls)} E-level balls to the vector database.")

    def add_a_propositions(self, propositions, embedding_model):
        if not propositions:
            return

        ids = [p["idx"] for p in propositions]
        docs = [p["proposition"] for p in propositions]
        embs = embedding_model.encode(docs).tolist()
        metas = [{"entity_list": ", ".join(p["entity_list"])} for p in propositions]

        batch_size = self.client.get_max_batch_size()
        total_batches = (len(ids) + batch_size - 1) // batch_size
        print(f"[Batch] {len(propositions)} A-level propositions, {batch_size} per batch, {total_batches} batches...")

        for i in range(0, len(ids), batch_size):
            self.a_collection.upsert(
                ids=ids[i:i + batch_size],
                embeddings=embs[i:i + batch_size],
                documents=docs[i:i + batch_size],
                metadatas=metas[i:i + batch_size],
            )
            print(f"[OK] A-level propositions written: {min(i + batch_size, len(ids))} / {len(ids)}")

        print(f"[OK] Persisted {len(propositions)} A-level propositions to the vector database.")


    def search_e_level(self, query_emb, top_k=3):
        res = self.c_collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
        return res['ids'][0] if res['ids'] else []

    def search_a_level(self, query_emb, top_k=8):
        res = self.a_collection.query(query_embeddings=[query_emb.tolist()], n_results=top_k)
        return res['ids'][0] if res['ids'] else []

def map_reduce_summary(idx_str, Propositions, llm, length):
    propositions = []

    for idx_item in idx_str:
        _id = int(idx_item.split('_')[-1])
        propositions.append(Propositions[_id]["proposition"])

    # Escape braces in proposition texts so PromptTemplate's str.format parsing
    # does not break (medical/encyclopedic texts often contain unpaired braces
    # such as "TGF-{beta}").
    safe_propositions = "\n".join(
        str(p).replace("{", "{{").replace("}", "}}") for p in propositions
    )
    prompt_str = f"""Based on the following set of domain-specific propositions, generate a complete and concise abstract of no more than {length} words:
    {safe_propositions}
    Requirements for the abstract:
    1. Focus strictly on the core information of each proposition without divergence;
    2. Retain key dimensions (time/location/core behavior/attributes);
    3. Fluent language, no redundancy, and consistent with the semantics of the domain to which the propositions belong.
    4. The abstract must be objective, without subjective evaluations, emotional adjectives or speculative content.
   """
    prompt = PromptTemplate(
        input_variables=["Propositions", "length"],
        template=prompt_str
    )

    chain = prompt | llm

    response = chain.invoke({"Propositions": propositions,
                           "length": length})
    summary = response.content
    # print('summary:')
    # print(summary)
    # print('\n')
    return summary


def build_full_graph(propositions, embedding_model, llm, vector_db_persist_dir="./graph_chroma_db"):
    """
    Build the complete granular-ball hierarchy graph (A/E/D + edges).

    Args:
        propositions: List[dict] - A-level proposition list, each item with:
            - idx: str, e.g. 'A_0'
            - proposition: str
            - entity_list: List[str]
            - extend_key_list: List[str]
            - time_entity_list: List[str]
        embedding_model: SentenceTransformer instance
        llm: ChatOllama / ChatOpenAI-compatible instance
        vector_db_persist_dir: str - vector database persistence directory

    Returns:
        tuple: (E_balls, D_balls, edges, db_helper)
            - E_balls: List[dict] - E-level (entity) ball list
            - D_balls: List[dict] - D-level (domain) ball list
            - edges: List[dict] - edge list
            - db_helper: VectorDBHelper - vector database helper (data written)
    """

    cache_dir = vector_db_persist_dir
    props_hash = hashlib.md5(
        json.dumps(propositions, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    cache_flag_path = os.path.join(cache_dir, "build_cache_flag.txt")

    if os.path.exists(cache_flag_path):
        try:
            with open(os.path.join(cache_dir, "props_hash.txt"), "r") as f:
                cached_hash = f.read().strip()
            if cached_hash == props_hash:
                print("[Cache] Cache is valid; skipping graph build and loading from cache...")
                E_balls = json.load(open(os.path.join(cache_dir, "E_balls.json"), "r"))
                edges = json.load(open(os.path.join(cache_dir, "edges.json"), "r"))
                db_helper = VectorDBHelper(persist_directory=vector_db_persist_dir)
                print("[OK] Cache loaded!")
                return E_balls, D_balls, edges, db_helper
        except Exception as e:
            print(f"[Warn] Cache load failed ({e}); rebuilding the graph...")

    print("=" * 60)
    print("Building the granular-ball hierarchy graph...")
    print("=" * 60)

    # Stage cache is keyed by the proposition hash, so interrupted builds resume safely.
    stage2 = _load_stage(cache_dir, props_hash, "entity_dict")
    if stage2 is None:
        print("[2/8] Counting entity frequencies...", flush=True)
        pre_entity_dic = pre_fun_count(propositions, embedding_model)
        entity_dict = func_count(pre_entity_dic)
        _save_stage(cache_dir, props_hash, "entity_dict", entity_dict)
    else:
        entity_dict = stage2
        print(f"[2/8] Reusing stage cache (entity clusters: {len(entity_dict)})", flush=True)
    N, entity_idx = make_sureN(entity_dict)
    print(f"    High-frequency entity threshold N = {N}")

    # 3. embed all propositions
    embeddings = _load_stage(cache_dir, props_hash, "embeddings")
    if embeddings is None:
        print("[3/8] Encoding proposition embeddings...", flush=True)
        texts = [item['proposition'] for item in propositions]
        embeddings = embedding_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()
        _save_stage(cache_dir, props_hash, "embeddings", embeddings)
    else:
        print("[3/8] Reusing stage cache (proposition embeddings)", flush=True)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    print(f"    {len(embeddings)} proposition vectors")

    # 4. build E-level balls (entity balls)
    stage4 = _load_stage(cache_dir, props_hash, "e_balls")
    if stage4 is None:
        print("[4/8] Building E-level balls...", flush=True)
        E_balls = build_e_balls(entity_dict, embeddings, N, entity_idx, propositions)
        _save_stage(cache_dir, props_hash, "e_balls", E_balls)
    else:
        E_balls = stage4
        print("[4/8] Reusing stage cache (E balls)", flush=True)
    print(f"    Initial E ball count: {len(E_balls)}")

    # 5. absorb unclustered proposition nodes into E balls
    stage5 = _load_stage(cache_dir, props_hash, "e_balls_unclustered")
    if stage5 is None:
        print("[5/8] Processing unclustered nodes...", flush=True)
        E_balls, one_node_list = unclustered_nodes_func(
            entity_dict, E_balls, embeddings, len(propositions), embedding_model, llm, propositions
        )
        _save_stage(cache_dir, props_hash, "e_balls_unclustered", {"e_balls": E_balls, "one_node_list": one_node_list})
    else:
        E_balls, one_node_list = stage5["e_balls"], stage5["one_node_list"]
        print("[5/8] Reusing stage cache (unclustered-node processing)", flush=True)
    print(f"    E ball count after processing: {len(E_balls)}; remaining isolated propositions: {len(one_node_list)}")

    # 6. split D-level balls (domain balls) out of the E balls
    stage6 = _load_stage(cache_dir, props_hash, "d_balls")
    if stage6 is None:
        print("[6/8] Splitting D-level balls...", flush=True)
        D_balls, E_balls = split_e_into_d_balls(E_balls, propositions, embeddings, embedding_model)
        _save_stage(cache_dir, props_hash, "d_balls", {"d_balls": D_balls, "e_balls": E_balls})
    else:
        D_balls, E_balls = stage6["d_balls"], stage6["e_balls"]
        print("[6/8] Reusing stage cache (D balls)", flush=True)
    print(f"    D ball count: {len(D_balls)}")

    # 7. build edges between E balls (based on boundary-node entity overlap)
    stage7 = _load_stage(cache_dir, props_hash, "edges")
    if stage7 is None:
        print("[7/8] Building edges between E balls...", flush=True)
        E_balls, D_balls, edges, edges_key = make_boundary_edges(E_balls, D_balls, propositions, embeddings)
        _save_stage(cache_dir, props_hash, "edges", {"e_balls": E_balls, "d_balls": D_balls, "edges": edges})
    else:
        E_balls, D_balls, edges = stage7["e_balls"], stage7["d_balls"], stage7["edges"]
        edges_key = set()
        print("[7/8] Reusing stage cache (edges)", flush=True)
    print(f"    Edge count: {len(edges)}")

    # 8. generate summaries for E balls
    stage8 = _load_stage(cache_dir, props_hash, "e_balls_summary")
    if stage8 is None:
        print("[8/8] Generating E-ball summaries...", flush=True)
        E_balls = summarize_e_balls(E_balls, llm, propositions)
        _save_stage(cache_dir, props_hash, "e_balls_summary", E_balls)
    else:
        E_balls = stage8
        print("[8/8] Reusing stage cache (E-ball summaries)", flush=True)

    # 9. write to the vector database
    print("\n[Extra] Writing to the vector database...")
    db_helper = VectorDBHelper(persist_directory=vector_db_persist_dir)
    db_helper.add_e_balls(E_balls, embedding_model)
    db_helper.add_a_propositions(propositions, embedding_model)
    print("    Vector database write complete!")

    print("\n" + "=" * 60)
    print("[OK] Granular-ball hierarchy graph built!")
    print(f"    A-level propositions: {len(propositions)}")
    print(f"    D-level balls: {len(D_balls)}")
    print(f"    E-level balls: {len(E_balls)}")
    print(f"    Edges: {len(edges)}")
    print("=" * 60)

    # ---------- save the cache ----------
    os.makedirs(cache_dir, exist_ok=True)
    json.dump(E_balls, open(os.path.join(cache_dir, "E_balls.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    json.dump(edges, open(os.path.join(cache_dir, "edges.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)
    with open(os.path.join(cache_dir, "props_hash.txt"), "w") as f:
        f.write(props_hash)
    with open(cache_flag_path, "w") as f:
        f.write("1")
    # ----------------------------


    return E_balls, D_balls, edges, db_helper


# ==========================================
# No-edges version of the graph builder (for ablation experiments)
# ==========================================
