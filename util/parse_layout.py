def parse_layout(layout_str):
    import re

    result = {
        'roll': None,
        'dim': []
    }

    # Extract roll numbers if present
    roll_match = re.search(r'roll\((\d+),(\d+)\)', layout_str)
    if roll_match:
        result['roll'] = {
            'first': int(roll_match.group(1)),
            'second': int(roll_match.group(2))
        }

    # Extract all bracket terms [number:number:number] or [number:number]
    bracket_matches = re.findall(r'\[\d+:\d+(?::\d+)?\]', layout_str)
    if bracket_matches:
        for bracket in bracket_matches:
            # Remove the brackets and split by colon
            result['dim'].append(bracket)

    return result


# Test with your examples
test_string1 = "[0:32:1][1:32:32][0:32:32];[1:32:1][128:1]"
test_string2 = "roll(0,3) [0:32:1][1:32:32][0:32:32];[1:32:1][128:1]"

print("Test 1:", parse_layout(test_string1))
print("Test 2:", parse_layout(test_string2))
