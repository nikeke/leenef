local utf8 = require("utf8")

local superscripts = {
  ["⁰"] = "0",
  ["¹"] = "1",
  ["²"] = "2",
  ["³"] = "3",
  ["⁴"] = "4",
  ["⁵"] = "5",
  ["⁶"] = "6",
  ["⁷"] = "7",
  ["⁸"] = "8",
  ["⁹"] = "9",
  ["⁻"] = "-",
}

local direct_replacements = {
  { "Â", "{\\hat{A}}" },
  { "ŝ", "{\\hat{s}}" },
  { "Σ", "{\\Sigma}" },
  { "σ", "{\\sigma}" },
  { "α", "{\\alpha}" },
  { "β", "{\\beta}" },
  { "γ", "{\\gamma}" },
  { "δ", "{\\delta}" },
  { "λ", "{\\lambda}" },
  { "μ", "{\\mu}" },
  { "π", "{\\pi}" },
  { "ℝ", "\\mathbb{R}" },
  { "ℤ", "\\mathbb{Z}" },
  { "ℕ", "\\mathbb{N}" },
  { "ᵀ", "^{\\top}" },
  { "·", " \\cdot " },
  { "×", " \\times " },
  { "∈", " \\in " },
  { "∼", " \\sim " },
  { "≈", " \\approx " },
  { "≤", " \\le " },
  { "≥", " \\ge " },
  { "±", " \\pm " },
  { "→", " \\to " },
  { "←", " \\leftarrow " },
  { "⊙", " \\odot " },
  { "…", "\\ldots{}" },
  { "−", "-" },
}

local math_symbols = "[ÂŝΣσαβγδλμπℝℤℕᵀ·×∈∼≈≤≥±→←⊙⁰¹²³⁴⁵⁶⁷⁸⁹⁻]"

local function trim(text)
  return (text:gsub("^%s+", ""):gsub("%s+$", ""))
end

local function replace_superscripts(text)
  local parts = {}
  local run = {}

  local function flush_run()
    if #run > 0 then
      parts[#parts + 1] = "^{" .. table.concat(run) .. "}"
      run = {}
    end
  end

  for _, codepoint in utf8.codes(text) do
    local ch = utf8.char(codepoint)
    local mapped = superscripts[ch]
    if mapped then
      run[#run + 1] = mapped
    else
      flush_run()
      parts[#parts + 1] = ch
    end
  end

  flush_run()
  return table.concat(parts)
end

local function is_short_subscript_symbol(text)
  local prefix = text:match("^([^_]+)_")
  return prefix ~= nil and utf8.len(prefix) ~= nil and utf8.len(prefix) <= 2
end

local function is_mathish(text)
  if text:find("\n") then
    return false
  end

  if text:match(math_symbols) then
    return true
  end

  if text:find("[=%+%-%*/%(%)%{%}%|]") and text:find("[%a_]") then
    return true
  end

  if is_short_subscript_symbol(text) then
    return true
  end

  return false
end

local function mathify(text)
  local note
  text = trim(text)
  text = text:gsub("([%aÂŝ])%*ᵀ", "%1@@STARTRANSPOSE@@")

  local main, trailing_note = text:match("^(.-)%s+%(([%a%- ]+)%)$")
  if main ~= nil and trailing_note ~= nil and main:find("=") then
    text = trim(main)
    note = trailing_note
  end

  text = replace_superscripts(text)
  text = text:gsub("([%aÂŝ}])%*", "%1^{*}")
  text = text:gsub("diag%s*%(", "\\operatorname{diag}(")
  text = text:gsub("trace%s*%(", "\\operatorname{tr}(")
  text = text:gsub("max%s*%(", "\\max(")

  for _, replacement in ipairs(direct_replacements) do
    text = text:gsub(replacement[1], replacement[2])
  end

  text = text:gsub("@@STARTRANSPOSE@@", "^{*\\top}")
  text = text:gsub("\\sim%s+training%s+set", "\\sim \\text{training set}")

  if note ~= nil then
    text = text .. " \\qquad \\text{(" .. note .. ")}"
  end

  return text
end

function Code(elem)
  if not is_mathish(elem.text) then
    return nil
  end

  return pandoc.Math("InlineMath", mathify(elem.text))
end

function CodeBlock(elem)
  if #elem.classes > 0 or not is_mathish(trim(elem.text)) then
    return nil
  end

  return pandoc.Para({ pandoc.Math("DisplayMath", mathify(elem.text)) })
end
