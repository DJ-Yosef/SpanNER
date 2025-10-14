import json

def convert_wyd_format(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    content = raw_data.get("text", "")
    entities = raw_data.get("data", {}).get("entity", [])

    result = {
        "text": content,
        "entities": []
    }

    for ent in entities:
        start, end = ent["position"]
        label = ent["definitionId"]
        entity_text = content[start:end]
        result["entities"].append({
            "start": start,
            "end": end,
            "label": label,
            "text": entity_text
        })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"转换完成：{output_path}")


if __name__ == "__main__":
       #  在这里修改要转换的文件路径
    input_json = "艺文类聚_兵書_1757244815970.wyd.json" 
    output_json = "converted_output.json"
    convert_wyd_format(input_json, output_json)
