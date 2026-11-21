--[[
Aegisub Automation 4 Macro: 3D Perspective Planar Tracker
Pairs with perspective_solver.py to track a 4-corner planar sign across video
frames on CPU and convert the 2D homography into exact frame-by-frame libass 3D
tags (\an5, \pos, \org, \fscx, \fscy, \frx, \fry, \frz).
]]

script_name = "3D Perspective Planar Tracker"
script_description = "Tracks a 4-corner \\clip planar sign across video frames and solves exact 3D ASS perspective tags."
script_author = "Perspective Tracker"
script_version = "1.0.0"

local AI_MESH_METHOD = "AI Mesh Import (Google Colab CoTracker3 JSON)"
local CPU_MESH_METHOD = "CPU Non-Rigid Mesh Optical Flow (Bending / Distort)"

local TRACKING_METHODS = {
    "Hybrid Optical Flow + Feature Homography (Recommended)",
    "Dense Corner Lucas-Kanade Optical Flow",
    "ORB Feature Matching + RANSAC",
    CPU_MESH_METHOD,
    AI_MESH_METHOD,
}

local TRACKING_DIRECTIONS = {
    "Forward (Start Frame -> End Frame)",
    "Backward (End Frame -> Start Frame)",
}

local ASPECT_MODES = {
    "Auto-Estimate Plane Aspect Ratio",
    "Keep Original Text Aspect Ratio",
}

local DISTORTION_MODES = {
    "Option A: Vector Drawing Mesh Warp (\\p4 Outlines)",
    "Option B: Per-Character 3D Tags (Native Font)",
    "Option C: Multi-Plane Fold Split + Dynamic \\clip",
}

local function is_windows()
    return package.config:sub(1, 1) == "\\"
end

local function path_sep()
    return is_windows() and "\\" or "/"
end

local function dirname(path)
    if not path or path == "" then return "" end
    local clean = path:gsub("[/\\]+$", "")
    local parent = clean:match("^(.*)[/\\][^/\\]+$")
    return parent or ""
end

local function join_path(a, b)
    if not a or a == "" then return b end
    local sep = path_sep()
    return a:gsub("[/\\]+$", "") .. sep .. b:gsub("^[/\\]+", "")
end

local function file_exists(path)
    if not path or path == "" then return false end
    local f = io.open(path, "rb")
    if f then
        f:close()
        return true
    end
    return false
end

local function show_error(msg)
    aegisub.dialog.display(
        {{class = "label", label = msg, x = 0, y = 0}},
        {"OK"},
        {ok = "OK", cancel = "OK"}
    )
end

-- ---------------------------------------------------------------------------
-- Pure-Lua JSON Codec (zero external dependencies)
-- ---------------------------------------------------------------------------

local json = {}

local function utf8_from_codepoint(cp)
    if cp < 0x80 then
        return string.char(cp)
    elseif cp < 0x800 then
        return string.char(
            0xC0 + math.floor(cp / 0x40),
            0x80 + (cp % 0x40)
        )
    elseif cp < 0x10000 then
        return string.char(
            0xE0 + math.floor(cp / 0x1000),
            0x80 + (math.floor(cp / 0x40) % 0x40),
            0x80 + (cp % 0x40)
        )
    else
        return string.char(
            0xF0 + math.floor(cp / 0x40000),
            0x80 + (math.floor(cp / 0x1000) % 0x40),
            0x80 + (math.floor(cp / 0x40) % 0x40),
            0x80 + (cp % 0x40)
        )
    end
end

local function json_escape_str(s)
    local out = s:gsub('[%z\1-\31\\"]', function(c)
        local b = string.byte(c)
        if c == "\\" then return "\\\\"
        elseif c == '"' then return '\\"'
        elseif c == "\b" then return "\\b"
        elseif c == "\f" then return "\\f"
        elseif c == "\n" then return "\\n"
        elseif c == "\r" then return "\\r"
        elseif c == "\t" then return "\\t"
        else return string.format("\\u%04x", b)
        end
    end)
    return '"' .. out .. '"'
end

local function is_array_table(t)
    if type(t) ~= "table" then return false end
    local max_idx = 0
    local count = 0
    for k, _ in pairs(t) do
        if type(k) ~= "number" or k < 1 or math.floor(k) ~= k then
            return false
        end
        if k > max_idx then max_idx = k end
        count = count + 1
    end
    return count > 0 and max_idx == count
end

local function encode_value(val, indent_level, pretty)
    local tv = type(val)
    if val == nil then
        return "null"
    elseif tv == "boolean" then
        return val and "true" or "false"
    elseif tv == "number" then
        if val ~= val or val == math.huge or val == -math.huge then
            return "null"
        end
        return tostring(val)
    elseif tv == "string" then
        return json_escape_str(val)
    elseif tv == "table" then
        local pad = pretty and string.rep("  ", indent_level) or ""
        local next_pad = pretty and string.rep("  ", indent_level + 1) or ""
        local nl = pretty and "\n" or ""
        local sp = pretty and " " or ""

        if is_array_table(val) then
            local parts = {}
            for i = 1, #val do
                parts[#parts + 1] = next_pad .. encode_value(val[i], indent_level + 1, pretty)
            end
            return "[" .. nl .. table.concat(parts, "," .. nl) .. nl .. pad .. "]"
        else
            local keys = {}
            for k in pairs(val) do
                keys[#keys + 1] = tostring(k)
            end
            table.sort(keys)
            if #keys == 0 then
                return "{}"
            end
            local parts = {}
            for _, k in ipairs(keys) do
                parts[#parts + 1] = next_pad .. json_escape_str(k) .. ":" .. sp .. encode_value(val[k], indent_level + 1, pretty)
            end
            return "{" .. nl .. table.concat(parts, "," .. nl) .. nl .. pad .. "}"
        end
    else
        error("Unsupported type for JSON encoding: " .. tv)
    end
end

function json.encode(val, pretty)
    return encode_value(val, 0, pretty == true)
end

function json.decode(str)
    if type(str) ~= "string" or str == "" then
        return nil, "Empty JSON input"
    end
    local pos = 1
    local len = #str

    local function skip_ws()
        local _, next_pos = str:find("^[ \t\r\n]+", pos)
        if next_pos then
            pos = next_pos + 1
        end
    end

    local parse_value

    local function parse_string()
        pos = pos + 1 -- skip opening quote
        local parts = {}
        while pos <= len do
            local c = str:sub(pos, pos)
            if c == '"' then
                pos = pos + 1
                return table.concat(parts)
            elseif c == "\\" then
                local esc = str:sub(pos + 1, pos + 1)
                if esc == '"' or esc == "\\" or esc == "/" then
                    parts[#parts + 1] = esc
                    pos = pos + 2
                elseif esc == "b" then
                    parts[#parts + 1] = "\b"; pos = pos + 2
                elseif esc == "f" then
                    parts[#parts + 1] = "\f"; pos = pos + 2
                elseif esc == "n" then
                    parts[#parts + 1] = "\n"; pos = pos + 2
                elseif esc == "r" then
                    parts[#parts + 1] = "\r"; pos = pos + 2
                elseif esc == "t" then
                    parts[#parts + 1] = "\t"; pos = pos + 2
                elseif esc == "u" then
                    local hex = str:sub(pos + 2, pos + 5)
                    local cp = tonumber(hex, 16)
                    if not cp then return nil, "Invalid unicode escape at pos " .. pos end
                    pos = pos + 6
                    -- Handle UTF-16 surrogate pairs
                    if cp >= 0xD800 and cp <= 0xDBFF and str:sub(pos, pos + 1) == "\\u" then
                        local hex2 = str:sub(pos + 2, pos + 5)
                        local cp2 = tonumber(hex2, 16)
                        if cp2 and cp2 >= 0xDC00 and cp2 <= 0xDFFF then
                            cp = 0x10000 + ((cp - 0xD800) * 0x400) + (cp2 - 0xDC00)
                            pos = pos + 6
                        end
                    end
                    parts[#parts + 1] = utf8_from_codepoint(cp)
                else
                    return nil, "Invalid escape sequence at pos " .. pos
                end
            else
                local nxt = str:find('[\\"]', pos)
                if not nxt then
                    return nil, "Unterminated JSON string"
                end
                parts[#parts + 1] = str:sub(pos, nxt - 1)
                pos = nxt
            end
        end
        return nil, "Unterminated JSON string"
    end

    local function parse_number()
        local s, e = str:find("^-?%d+[%.%deE+-]*", pos)
        if not s then
            return nil, "Invalid number at pos " .. pos
        end
        local num = tonumber(str:sub(s, e))
        if not num then
            return nil, "Malformed number at pos " .. pos
        end
        pos = e + 1
        return num
    end

    local function parse_array()
        pos = pos + 1 -- skip '['
        skip_ws()
        local arr = {}
        if str:sub(pos, pos) == "]" then
            pos = pos + 1
            return arr
        end
        while true do
            local val, err = parse_value()
            if err then return nil, err end
            arr[#arr + 1] = val
            skip_ws()
            local ch = str:sub(pos, pos)
            if ch == "]" then
                pos = pos + 1
                return arr
            elseif ch == "," then
                pos = pos + 1
            else
                return nil, "Expected ',' or ']' in array at pos " .. pos
            end
        end
    end

    local function parse_object()
        pos = pos + 1 -- skip '{'
        skip_ws()
        local obj = {}
        if str:sub(pos, pos) == "}" then
            pos = pos + 1
            return obj
        end
        while true do
            skip_ws()
            if str:sub(pos, pos) ~= '"' then
                return nil, "Expected object key string at pos " .. pos
            end
            local key, kerr = parse_string()
            if kerr then return nil, kerr end
            skip_ws()
            if str:sub(pos, pos) ~= ":" then
                return nil, "Expected ':' after object key at pos " .. pos
            end
            pos = pos + 1
            local val, verr = parse_value()
            if verr then return nil, verr end
            obj[key] = val
            skip_ws()
            local ch = str:sub(pos, pos)
            if ch == "}" then
                pos = pos + 1
                return obj
            elseif ch == "," then
                pos = pos + 1
            else
                return nil, "Expected ',' or '}' in object at pos " .. pos
            end
        end
    end

    parse_value = function()
        skip_ws()
        if pos > len then
            return nil, "Unexpected end of JSON input"
        end
        local ch = str:sub(pos, pos)
        if ch == '"' then
            return parse_string()
        elseif ch == "{" then
            return parse_object()
        elseif ch == "[" then
            return parse_array()
        elseif ch == "-" or (ch >= "0" and ch <= "9") then
            return parse_number()
        elseif str:sub(pos, pos + 3) == "true" then
            pos = pos + 4
            return true
        elseif str:sub(pos, pos + 4) == "false" then
            pos = pos + 5
            return false
        elseif str:sub(pos, pos + 3) == "null" then
            pos = pos + 4
            return nil
        else
            return nil, "Unexpected character '" .. ch .. "' at pos " .. pos
        end
    end

    local res, err = parse_value()
    if err then return nil, err end
    return res, nil
end

-- ---------------------------------------------------------------------------
-- Config Management (%appdata%/aegisub/config/perspective_tracker.json & Linux)
-- ---------------------------------------------------------------------------

local function get_config_path()
    if is_windows() then
        local appdata = os.getenv("APPDATA")
        if appdata and appdata ~= "" then
            return join_path(join_path(appdata, "aegisub"), "config\\perspective_tracker.json")
        end
        return aegisub.decode_path("?user/config/perspective_tracker.json")
    else
        local xdg = os.getenv("XDG_CONFIG_HOME")
        if xdg and xdg ~= "" then
            return join_path(join_path(xdg, "aegisub"), "config/perspective_tracker.json")
        end
        local home = os.getenv("HOME")
        if home and home ~= "" then
            return join_path(join_path(home, ".config/aegisub"), "config/perspective_tracker.json")
        end
        return aegisub.decode_path("?user/config/perspective_tracker.json")
    end
end

local function ensure_parent_dir(file_path)
    local dir = dirname(file_path)
    if dir == "" then return end
    if is_windows() then
        os.execute('cmd /c if not exist "' .. dir .. '" mkdir "' .. dir .. '"')
    else
        os.execute('mkdir -p "' .. dir:gsub('"', '\\"') .. '"')
    end
end

local function detect_default_solver_path()
    local candidates = {}

    -- 1. Next to this Lua script
    if debug and debug.getinfo then
        local info = debug.getinfo(1, "S")
        if info and info.source and info.source:sub(1, 1) == "@" then
            local script_dir = dirname(info.source:sub(2))
            if script_dir ~= "" then
                candidates[#candidates + 1] = join_path(script_dir, "perspective_solver.py")
            end
        end
    end

    -- 2. Standard Aegisub automation paths
    local aegi_paths = {
        "?user/automation/autoload/perspective_solver.py",
        "?user/automation/perspective_solver.py",
        "?data/automation/autoload/perspective_solver.py",
        "?script/perspective_solver.py",
    }
    for _, ap in ipairs(aegi_paths) do
        local decoded = aegisub.decode_path(ap)
        if decoded and decoded ~= "" then
            candidates[#candidates + 1] = decoded
        end
    end

    for _, cand in ipairs(candidates) do
        if file_exists(cand) then
            return cand
        end
    end
    return candidates[1] or "perspective_solver.py"
end

local function default_config()
    return {
        python_executable = is_windows() and "python" or "python3",
        python_script_path = detect_default_solver_path(),
        mesh_json_path = "",
        font_file_path = "",
        tracking_method = TRACKING_METHODS[1],
        tracking_direction = TRACKING_DIRECTIONS[1],
        aspect_ratio_mode = ASPECT_MODES[1],
        distortion_mode = DISTORTION_MODES[1],
        arabic_text = false,
        fov_factor = 1.0,
        smoothing_window = 0,
        mesh_cols = 10,
        mesh_rows = 6,
    }
end

local function in_list(val, list)
    for _, item in ipairs(list) do
        if item == val then return true end
    end
    return false
end

local function load_config()
    local cfg = default_config()
    local path = get_config_path()
    local f = io.open(path, "rb")
    if not f then
        return cfg
    end
    local raw = f:read("*a")
    f:close()

    local parsed = json.decode(raw)
    if type(parsed) ~= "table" then
        return cfg
    end

    if type(parsed.python_executable) == "string" and parsed.python_executable ~= "" then
        cfg.python_executable = parsed.python_executable
    end
    if type(parsed.python_script_path) == "string" and parsed.python_script_path ~= "" then
        cfg.python_script_path = parsed.python_script_path
    end
    if type(parsed.mesh_json_path) == "string" then
        cfg.mesh_json_path = parsed.mesh_json_path
    end
    if type(parsed.font_file_path) == "string" then
        cfg.font_file_path = parsed.font_file_path
    end
    if type(parsed.tracking_method) == "string" and in_list(parsed.tracking_method, TRACKING_METHODS) then
        cfg.tracking_method = parsed.tracking_method
    end
    if type(parsed.tracking_direction) == "string" and in_list(parsed.tracking_direction, TRACKING_DIRECTIONS) then
        cfg.tracking_direction = parsed.tracking_direction
    end
    if type(parsed.aspect_ratio_mode) == "string" and in_list(parsed.aspect_ratio_mode, ASPECT_MODES) then
        cfg.aspect_ratio_mode = parsed.aspect_ratio_mode
    end
    if type(parsed.distortion_mode) == "string" and in_list(parsed.distortion_mode, DISTORTION_MODES) then
        cfg.distortion_mode = parsed.distortion_mode
    elseif parsed.opt_per_char == true then
        cfg.distortion_mode = DISTORTION_MODES[2]
    elseif parsed.opt_fold_split == true then
        cfg.distortion_mode = DISTORTION_MODES[3]
    end
    if type(parsed.arabic_text) == "boolean" then
        cfg.arabic_text = parsed.arabic_text
    end
    if type(parsed.fov_factor) == "number" and parsed.fov_factor > 0.05 then
        cfg.fov_factor = parsed.fov_factor
    end
    if type(parsed.smoothing_window) == "number" then
        cfg.smoothing_window = math.max(0, math.min(10, math.floor(parsed.smoothing_window + 0.5)))
    end
    if type(parsed.mesh_cols) == "number" then
        cfg.mesh_cols = math.max(3, math.min(24, math.floor(parsed.mesh_cols + 0.5)))
    end
    if type(parsed.mesh_rows) == "number" then
        cfg.mesh_rows = math.max(3, math.min(24, math.floor(parsed.mesh_rows + 0.5)))
    end

    return cfg
end

local function save_config(cfg)
    local path = get_config_path()
    local f, err = io.open(path, "wb")
    if not f then
        ensure_parent_dir(path)
        f, err = io.open(path, "wb")
    end
    if not f then
        return false, err
    end
    f:write(json.encode(cfg, true) .. "\n")
    f:close()
    return true, nil
end

-- ---------------------------------------------------------------------------
-- ASS Metadata, Clip Extraction & Validation
-- ---------------------------------------------------------------------------

local function validate_and_extract_clip(line_text)
    local payload = line_text:match("\\i?clip%(([^%)]*)%)")
    if not payload then
        return nil, "The selected guide line does not contain a \\clip(...) tag.\n\n"
            .. "Please use Aegisub's Vector Clip tool to draw a 4-corner polygon:\n"
            .. "\\clip(m x1 y1 l x2 y2 x3 y3 x4 y4) in clockwise order (TL, TR, BR, BL)."
    end

    local trim = payload:match("^%s*(.-)%s*$")
    local scale_denom = 1.0
    if trim:find(",", 1, true) then
        local first, rest = trim:match("^([^,]+),(.+)$")
        if first and rest then
            local num = tonumber(first:match("^%s*(.-)%s*$"))
            if num and rest:find("[mMlL]") then
                scale_denom = 2 ^ (num - 1)
                trim = rest
            else
                return nil, "Rectangular \\clip(x1,y1,x2,y2) is not supported.\n\n"
                    .. "Please draw a 4-point Vector Clip (m x1 y1 l x2 y2 x3 y3 x4 y4)."
            end
        end
    end

    if trim:find("[bBsSpPcC]") then
        return nil, "Bezier/spline commands in \\clip are not supported.\n\n"
            .. "Please use straight line segments (m ... l ...) with exactly 4 corners."
    end

    local coords = {}
    local cmd = nil
    local i = 1
    local len = #trim
    while i <= len do
        local ch = trim:sub(i, i)
        if ch:match("%s") then
            i = i + 1
        elseif ch:match("%a") then
            local s, e = trim:find("^%a+", i)
            cmd = trim:sub(s, e):lower()
            if cmd ~= "m" and cmd ~= "n" and cmd ~= "l" then
                return nil, "Unsupported drawing command '" .. cmd .. "' in \\clip. Use only 'm' and 'l'."
            end
            i = e + 1
        else
            local s, e = trim:find("^[+-]?%d*%.?%d+", i)
            if not s then
                return nil, "Invalid coordinate syntax in \\clip: '" .. trim:sub(i) .. "'."
            end
            local val = tonumber(trim:sub(s, e))
            if not val then
                return nil, "Invalid numeric value in \\clip: '" .. trim:sub(s, e) .. "'."
            end
            if not cmd then
                return nil, "Invalid vector \\clip syntax (coordinates before 'm' command)."
            end
            coords[#coords + 1] = val / scale_denom
            i = e + 1
        end
    end

    if #coords % 2 ~= 0 then
        return nil, "Odd number of coordinate values in \\clip; every corner needs an (x, y) pair."
    end

    local pts = {}
    for j = 1, #coords, 2 do
        pts[#pts + 1] = {coords[j], coords[j + 1]}
    end

    if #pts == 5 then
        local dx = pts[5][1] - pts[1][1]
        local dy = pts[5][2] - pts[1][2]
        if math.sqrt(dx * dx + dy * dy) <= 1.0 then
            table.remove(pts, 5)
        end
    end

    if #pts ~= 4 then
        return nil, string.format(
            "Expected exactly 4 \\clip corners, but found %d.\n\n"
            .. "Please mark the 4 corners of the target sign plane in clockwise order "
            .. "(Top-Left, Top-Right, Bottom-Right, Bottom-Left).",
            #pts
        )
    end

    return pts, nil
end

local function extract_script_info_and_video(subtitles)
    local play_res_x = 0
    local play_res_y = 0
    local garbage_video = nil

    for i = 1, #subtitles do
        local line = subtitles[i]
        if line.class == "info" then
            if line.key == "PlayResX" then
                play_res_x = tonumber(line.value) or 0
            elseif line.key == "PlayResY" then
                play_res_y = tonumber(line.value) or 0
            elseif line.key == "Video File" then
                garbage_video = line.value
            end
        end
    end

    local vid_w, vid_h = aegisub.video_size()
    if play_res_x <= 0 and vid_w then play_res_x = vid_w end
    if play_res_y <= 0 and vid_h then play_res_y = vid_h end

    local props = aegisub.project_properties and aegisub.project_properties() or nil
    local video_path = (props and props.video_file and props.video_file ~= "") and props.video_file or garbage_video

    if not video_path or video_path == "" then
        return "", play_res_x, play_res_y, "No active video file is loaded in Aegisub.\n\nPlease open a video file (Video -> Open Video...) before running the tracker."
    end
    if video_path:match("^%?dummy") then
        return "", play_res_x, play_res_y, "A dummy video is currently loaded.\n\nPlease open an actual video file before running video tracking."
    end
    if video_path:sub(1, 1) == "?" then
        video_path = aegisub.decode_path(video_path)
    end

    -- Resolve relative paths against the script directory (?script).
    local is_abs = video_path:match("^[/\\]") or video_path:match("^%a:[/\\]")
    if not is_abs then
        local script_dir = aegisub.decode_path("?script/")
        if script_dir and script_dir ~= "" and not script_dir:match("^%?script") then
            video_path = join_path(script_dir, video_path)
        end
    end

    return video_path, play_res_x, play_res_y, nil
end

local function extract_style_info(subtitles, style_name)
    for i = 1, #subtitles do
        local line = subtitles[i]
        if line.class == "style" and line.name == style_name then
            return {
                fontname = line.fontname or "Arial",
                fontsize = tonumber(line.fontsize) or 48.0,
                bold = line.bold == true,
                italic = line.italic == true,
                spacing = tonumber(line.spacing) or 0.0,
                scale_x = tonumber(line.scale_x) or 100.0,
                scale_y = tonumber(line.scale_y) or 100.0,
            }
        end
    end
    return {
        fontname = "Arial",
        fontsize = 48.0,
        bold = false,
        italic = false,
        spacing = 0.0,
        scale_x = 100.0,
        scale_y = 100.0,
    }
end

-- ---------------------------------------------------------------------------
-- Subprocess Execution (io.popen)
-- ---------------------------------------------------------------------------

local function temp_json_path()
    local base = aegisub.decode_path("?temp")
    if not base or base == "" or base:sub(1, 1) == "?" then
        base = os.getenv("TEMP") or os.getenv("TMP") or "/tmp"
    end
    local unique = string.format("aegisub_persp_%d_%d.json", os.time(), math.random(10000, 99999))
    return join_path(base, unique)
end

local function invoke_python_solver(py_exe, py_script, request_payload)
    local req_path = temp_json_path()
    local f, werr = io.open(req_path, "wb")
    if not f then
        return nil, "Failed to write temporary request JSON to '" .. req_path .. "': " .. tostring(werr)
    end
    f:write(json.encode(request_payload, false))
    f:close()

    local cmd
    if is_windows() then
        -- Use `call` so Windows cmd.exe /c invoked by MSVCRT _popen never strips
        -- quotes from paths containing spaces (both python.exe and script paths).
        local exe_arg = py_exe:match("^py%s+%-") and py_exe or ('"' .. py_exe .. '"')
        cmd = 'call ' .. exe_arg .. ' "' .. py_script .. '" --input "' .. req_path .. '" 2>&1'
    else
        local function sh_quote(s)
            return "'" .. s:gsub("'", "'\\''") .. "'"
        end
        cmd = sh_quote(py_exe) .. " " .. sh_quote(py_script) .. " --input " .. sh_quote(req_path) .. " 2>&1"
    end

    local pipe, perr = io.popen(cmd, "r")
    if not pipe then
        os.remove(req_path)
        return nil, "Failed to invoke Python solver via io.popen: " .. tostring(perr)
    end

    local raw_out = pipe:read("*a") or ""
    pipe:close()
    os.remove(req_path)

    -- Extract JSON object in case any leading/trailing whitespace or diagnostic line appeared.
    local json_start = raw_out:find("{", 1, true)
    local json_str = json_start and raw_out:sub(json_start) or raw_out
    local decoded, derr = json.decode(json_str)
    if not decoded then
        return nil, "Python solver returned invalid output.\n\nCommand:\n" .. cmd .. "\n\nOutput:\n" .. raw_out .. "\n\nParse error: " .. tostring(derr)
    end

    if decoded.status ~= "ok" then
        return nil, decoded.message or "Unknown error returned by perspective_solver.py."
    end

    return decoded, nil
end

-- ---------------------------------------------------------------------------
-- GUI Dialog & Main Macro Entry Point
-- ---------------------------------------------------------------------------

local function show_tracker_dialog(cfg)
    while true do
        local dialog_config = {
            {class = "label", label = "Tracking Method:", x = 0, y = 0, width = 1, height = 1},
            {
                class = "dropdown", name = "tracking_method",
                items = TRACKING_METHODS, value = cfg.tracking_method,
                x = 1, y = 0, width = 2, height = 1,
            },
            {class = "label", label = "Tracking Direction:", x = 0, y = 1, width = 1, height = 1},
            {
                class = "dropdown", name = "tracking_direction",
                items = TRACKING_DIRECTIONS, value = cfg.tracking_direction,
                x = 1, y = 1, width = 2, height = 1,
            },
            {class = "label", label = "Aspect Ratio Mode:", x = 0, y = 2, width = 1, height = 1},
            {
                class = "dropdown", name = "aspect_ratio_mode",
                items = ASPECT_MODES, value = cfg.aspect_ratio_mode,
                x = 1, y = 2, width = 2, height = 1,
            },
            {class = "label", label = "Virtual Focal Length (FOV factor):", x = 0, y = 3, width = 1, height = 1},
            {
                class = "floatedit", name = "fov_factor",
                value = cfg.fov_factor, min = 0.1, max = 10.0, step = 0.1,
                x = 1, y = 3, width = 2, height = 1,
                hint = "Default 1.0 (~50mm lens relative to video width) for accurate \\frx/\\fry foreshortening",
            },
            {class = "label", label = "Smoothing Window (Frames):", x = 0, y = 4, width = 1, height = 1},
            {
                class = "intedit", name = "smoothing_window",
                value = cfg.smoothing_window, min = 0, max = 10,
                x = 1, y = 4, width = 2, height = 1,
                hint = "Temporal Gaussian smoothing radius (0 = raw per-frame, up to 10 frames)",
            },
            {class = "label", label = "CPU Mesh Grid (Cols x Rows):", x = 0, y = 5, width = 1, height = 1},
            {
                class = "intedit", name = "mesh_cols",
                value = cfg.mesh_cols, min = 3, max = 24,
                x = 1, y = 5, width = 1, height = 1,
                hint = "Mesh grid columns for CPU Non-Rigid Mesh mode (default 10)",
            },
            {
                class = "intedit", name = "mesh_rows",
                value = cfg.mesh_rows, min = 3, max = 24,
                x = 2, y = 5, width = 1, height = 1,
                hint = "Mesh grid rows for CPU Non-Rigid Mesh mode (default 6)",
            },
            {class = "label", label = "Distortion Mode (Mesh / AI):", x = 0, y = 6, width = 1, height = 1},
            {
                class = "dropdown", name = "distortion_mode",
                items = DISTORTION_MODES, value = cfg.distortion_mode,
                x = 1, y = 6, width = 2, height = 1,
                hint = "Single-choice output mode for CPU Non-Rigid Mesh and AI Mesh Import",
            },
            {
                class = "checkbox", name = "arabic_text",
                label = "Arabic Text (Join Letters & Invert RTL Visual Order)",
                value = cfg.arabic_text,
                x = 0, y = 7, width = 3, height = 1,
                hint = "Shapes Arabic text with OpenType / Presentation Forms-B and reverses RTL order for mesh warping",
            },
            {class = "label", label = "Font File (.ttf/.otf/.ttc):", x = 0, y = 8, width = 1, height = 1},
            {
                class = "edit", name = "font_file_path",
                text = cfg.font_file_path,
                x = 1, y = 8, width = 2, height = 1,
                hint = "Font file for Option A/B vector outlines (supports fonts loaded via NexusFont)",
            },
            {class = "label", label = "AI Mesh JSON (Colab):", x = 0, y = 9, width = 1, height = 1},
            {
                class = "edit", name = "mesh_json_path",
                text = cfg.mesh_json_path,
                x = 1, y = 9, width = 2, height = 1,
                hint = "Path to mesh_track.json exported from the Google Colab CoTracker3 notebook",
            },
            {class = "label", label = "Python Solver (.py) Path:", x = 0, y = 10, width = 1, height = 1},
            {
                class = "edit", name = "python_script_path",
                text = cfg.python_script_path,
                x = 1, y = 10, width = 2, height = 1,
            },
        }

        local buttons = {"Track & Solve", "Choose Font...", "Choose Mesh JSON...", "Choose .py Script...", "Cancel"}
        local btn, res = aegisub.dialog.display(
            dialog_config,
            buttons,
            {ok = "Track & Solve", cancel = "Cancel"}
        )

        if not btn or btn == "Cancel" then
            return nil
        end

        -- Persist current field values into cfg across picker round-trips.
        cfg.tracking_method = res.tracking_method
        cfg.tracking_direction = res.tracking_direction
        cfg.aspect_ratio_mode = res.aspect_ratio_mode
        cfg.distortion_mode = res.distortion_mode
        cfg.arabic_text = res.arabic_text == true
        cfg.fov_factor = tonumber(res.fov_factor) or 1.0
        cfg.smoothing_window = math.max(0, math.min(10, math.floor((tonumber(res.smoothing_window) or 0) + 0.5)))
        cfg.mesh_cols = math.max(3, math.min(24, math.floor((tonumber(res.mesh_cols) or 10) + 0.5)))
        cfg.mesh_rows = math.max(3, math.min(24, math.floor((tonumber(res.mesh_rows) or 6) + 0.5)))
        if type(res.font_file_path) == "string" then
            cfg.font_file_path = res.font_file_path
        end
        if type(res.mesh_json_path) == "string" then
            cfg.mesh_json_path = res.mesh_json_path
        end
        if res.python_script_path and res.python_script_path ~= "" then
            cfg.python_script_path = res.python_script_path
        end

        if btn == "Choose Font..." then
            local start_dir = cfg.font_file_path ~= "" and cfg.font_file_path or aegisub.decode_path("?script/")
            local chosen = aegisub.dialog.open(
                "Select Font File (.ttf / .otf / .ttc)",
                start_dir,
                "",
                "Font files (*.ttf;*.otf;*.ttc;*.otc)|*.ttf;*.otf;*.ttc;*.otc|All files (*.*)|*.*",
                false,
                true
            )
            if chosen and chosen ~= "" then
                cfg.font_file_path = chosen
                save_config(cfg)
            end
        elseif btn == "Choose Mesh JSON..." then
            local start_dir = cfg.mesh_json_path ~= "" and cfg.mesh_json_path or aegisub.decode_path("?script/")
            local chosen = aegisub.dialog.open(
                "Select Exported Colab Mesh JSON",
                start_dir,
                "",
                "JSON files (*.json)|*.json|All files (*.*)|*.*",
                false,
                true
            )
            if chosen and chosen ~= "" then
                cfg.mesh_json_path = chosen
                cfg.tracking_method = AI_MESH_METHOD
                save_config(cfg)
            end
        elseif btn == "Choose .py Script..." then
            local chosen = aegisub.dialog.open(
                "Select perspective_solver.py",
                cfg.python_script_path,
                "",
                "Python scripts (*.py)|*.py|All files (*.*)|*.*",
                false,
                true
            )
            if chosen and chosen ~= "" then
                cfg.python_script_path = chosen
                save_config(cfg)
            end
        elseif btn == "Track & Solve" then
            return cfg
        end
    end
end

local function clone_line(line)
    local copy = {}
    for k, v in pairs(line) do
        copy[k] = v
    end
    return copy
end

local function run_perspective_tracker(subtitles, selected_lines)
    if #selected_lines ~= 1 then
        show_error("Please select exactly one guide line before running the 3D Perspective Tracker.")
        aegisub.cancel()
        return
    end

    local guide_idx = selected_lines[1]
    local guide_line = subtitles[guide_idx]

    if guide_line.end_time <= guide_line.start_time then
        show_error("The selected guide line must have a positive duration spanning at least 1 frame.")
        aegisub.cancel()
        return
    end

    local video_path, play_res_x, play_res_y, vid_err = extract_script_info_and_video(subtitles)

    local cfg = load_config()
    cfg = show_tracker_dialog(cfg)
    if not cfg then
        aegisub.cancel()
        return
    end

    if not file_exists(cfg.python_script_path) then
        show_error(
            "Could not find perspective_solver.py at:\n"
            .. tostring(cfg.python_script_path)
            .. "\n\nPlease click 'Choose .py Script...' to select perspective_solver.py."
        )
        aegisub.cancel()
        return
    end

    local is_ai_mesh = cfg.tracking_method == AI_MESH_METHOD
    local is_cpu_mesh = cfg.tracking_method == CPU_MESH_METHOD
    local needs_font_file = (is_ai_mesh or is_cpu_mesh) and (cfg.distortion_mode ~= DISTORTION_MODES[3])

    if needs_font_file and (cfg.font_file_path == "" or not file_exists(cfg.font_file_path)) then
        show_error(
            "Font file not selected or not found:\n"
            .. tostring(cfg.font_file_path)
            .. "\n\nPlease click 'Choose Font...' to pick the .ttf / .otf / .ttc font file used for your sign."
        )
        aegisub.cancel()
        return
    end

    local clip_corners = nil
    if is_ai_mesh then
        if cfg.mesh_json_path == "" or not file_exists(cfg.mesh_json_path) then
            show_error(
                "AI Mesh JSON file not found:\n"
                .. tostring(cfg.mesh_json_path)
                .. "\n\nPlease click 'Choose Mesh JSON...' to select the mesh_track.json exported from Colab."
            )
            aegisub.cancel()
            return
        end
        local optional_clip = validate_and_extract_clip(guide_line.text)
        clip_corners = optional_clip
    else
        if vid_err then
            show_error(vid_err)
            aegisub.cancel()
            return
        end
        local clip_err
        clip_corners, clip_err = validate_and_extract_clip(guide_line.text)
        if not clip_corners then
            show_error(clip_err)
            aegisub.cancel()
            return
        end
    end

    -- Save configuration when applying script.
    save_config(cfg)

    local start_frame = aegisub.frame_from_ms(guide_line.start_time)
    local end_frame_excl = aegisub.frame_from_ms(guide_line.end_time)
    local end_frame = math.max(start_frame, end_frame_excl - 1)

    aegisub.progress.title(script_name)
    aegisub.progress.task(string.format("Solving frames %d to %d...", start_frame, end_frame))
    aegisub.progress.set(15)

    local request_payload = {
        video_path = video_path,
        start_frame = start_frame,
        end_frame = end_frame,
        play_res_x = play_res_x,
        play_res_y = play_res_y,
        clip_corners = clip_corners,
        text = guide_line.text,
        style = extract_style_info(subtitles, guide_line.style),
        tracking_method = cfg.tracking_method,
        tracking_direction = cfg.tracking_direction,
        aspect_ratio_mode = cfg.aspect_ratio_mode,
        distortion_mode = cfg.distortion_mode,
        arabic_text = cfg.arabic_text,
        font_file_path = cfg.font_file_path,
        fov_factor = cfg.fov_factor,
        smoothing_window = cfg.smoothing_window,
        mesh_json_path = cfg.mesh_json_path,
        mesh_cols = cfg.mesh_cols,
        mesh_rows = cfg.mesh_rows,
    }

    local result, solver_err = invoke_python_solver(
        cfg.python_executable,
        cfg.python_script_path,
        request_payload
    )
    if not result then
        show_error("Tracking failed:\n\n" .. tostring(solver_err))
        aegisub.cancel()
        return
    end

    local frames = result.frames or {}
    if #frames == 0 then
        show_error("The solver returned zero tracked frames.")
        aegisub.cancel()
        return
    end

    aegisub.progress.task("Inserting frame-by-frame ASS lines...")
    aegisub.progress.set(85)

    local insert_pos = guide_idx
    local new_selection = {}
    for _, frame_data in ipairs(frames) do
        local f_num = frame_data.frame
        local new_line = clone_line(guide_line)
        new_line.class = "dialogue"
        new_line.comment = false
        new_line.start_time = aegisub.ms_from_frame(f_num)
        new_line.end_time = aegisub.ms_from_frame(f_num + 1)
        new_line.text = frame_data.text
        if type(frame_data.effect) == "string" and frame_data.effect ~= "" then
            new_line.effect = frame_data.effect
        end
        subtitles.insert(insert_pos, new_line)
        new_selection[#new_selection + 1] = insert_pos
        insert_pos = insert_pos + 1
    end

    -- Comment out the original guide line (now shifted to insert_pos).
    local updated_guide = clone_line(guide_line)
    updated_guide.comment = true
    subtitles[insert_pos] = updated_guide

    aegisub.progress.set(100)
    aegisub.set_undo_point(script_name)
    return new_selection
end

local function validate_macro(subtitles, selected_lines)
    return #selected_lines > 0
end

aegisub.register_macro(script_name, script_description, run_perspective_tracker, validate_macro)
