from SimpleLogic.derivation_new import backderive_nextlayer_rules, ConjunctionRule
from SimpleLogic import ruleset

y = ruleset.RuleNode(word="y", rules=frozenset({
    frozenset({"y", "not f", "not g"}),          # f & g -> y
    frozenset({"y", "not a", "not g"}),          # a & g -> y
    frozenset({"y", "not a", "not e", "not g"}), # a & e & g -> y
}))

a = ruleset.RuleNode(word="a", rules=frozenset({
    frozenset({"a", "not b"}),                   # b -> a
}))

b = ruleset.RuleNode(word="b", rules=frozenset({
    frozenset({"b", "not d"}),                   # d -> b
    frozenset({"b", "not c", "not f"}),          # c & f -> b
}))

c = ruleset.RuleNode(word="c", rules=frozenset({
    frozenset({"c", "not d", "not e"}),          # d & e -> c
}))

d = ruleset.RuleNode(word="d", rules=frozenset({
    frozenset({"d", "not c"}),                   # c -> d
    frozenset({"d", "not a", "not b", "not f"}), # a & b & f -> d
}))

e = ruleset.RuleNode(word="e", rules=frozenset({
    frozenset({"e", "not b"}),                   # b -> e
    frozenset({"e", "not c", "not g"}),          # c & g -> e
}))

f = ruleset.RuleNode(word="f", rules=frozenset({
    frozenset({"f", "not c"}),                   # c -> f
    frozenset({"f", "not a", "not d", "not g"}), # a & d & g -> f
    frozenset({"f", "not d", "not e"}),          # d & e -> f
}))

g = ruleset.RuleNode(word="g", rules=frozenset({
    frozenset({"g", "not a", "not c", "not f"}), # a & c & f -> g
}))

rt = ruleset.RuleTree(nodes={"y":y,"a":a,"b":b,"c":c,"d":d,"e":e,"f":f,"g":g})

temp = backderive_nextlayer_rules(rt, {ConjunctionRule({"y":0}, {}, [], "y", 0)}, set(), 5)
all_rules = temp[1]
print(sorted([sorted(r.leaf_set) for r in all_rules if r.root_word == "y"]))
