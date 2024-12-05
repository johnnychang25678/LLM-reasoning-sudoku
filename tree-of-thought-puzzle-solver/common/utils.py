import json


def extract_json_from_text_string(text_str: str):
    try:
        lp_idx = text_str.index('{')
        rp_idx = text_str.rindex('}')
        json_str = text_str[lp_idx:rp_idx+1]
        # print("extract_json_from_text_string", json_str)
        json_str = json_str.replace("'", '"')
        print("extract_json_from_text_string", json_str)
        json_obj = json.loads(json_str)
        return True, json_obj
    except Exception as e:
        print("extract_json_from_text_string error", e)
        return False, None
