"""What every name in a bot's explanation means, served at `GET /api/glossary`.

The names are the scorer's own (`crates/ttr-core/src/heuristic/policy.rs` reports them
through its trace sink); a test plays every tier on both maps and fails on any name not
listed here, so the review screen never shows one it cannot explain.
"""

from __future__ import annotations

from ticket_to_ride.ui.web import protocol as p

#: Additive terms of a candidate's total, in points.
TERMS = {
    "route_points": "Points the route scores when claimed.",
    "train_cost": "What the train cars spent are worth elsewhere: every car costs a minimum "
    "number of points, so a claim must pay at least that per car.",
    "plan_value": "Progress toward held tickets: the cars this claim lays on the planned "
    "network, at what a car of the plan is worth.",
    "contest": "How much more the plan would cost if someone else took this segment.",
    "double_lockout": "Taking one track of a double route closes the other to everyone at "
    "two or three players, so a planned double is worth taking early.",
    "longest_chain": "Extends a network that already exists: a tiebreak toward the "
    "longest-route bonus.",
    "gray_cannibalization": "Paying a gray route with a colour a coloured planned route "
    "still needs.",
    "locomotive_thrift": "Spending locomotives where coloured cards would do.",
    "tickets_finished": "Its last turn: held tickets this claim completes, each swinging from "
    "minus its points to plus.",
    "endgame": "Late in the game a plan that can still be finished is urgent, and one that "
    "cannot is worth nothing.",
    "threat": "H4: denies an opponent the route that would join two of their networks.",
    "hoard": "H4: this planned segment is safe to leave for now while a contested one needs "
    "the cards.",
    "expected_card_value": "What a card from the deck is worth on average, over the cards "
    "this seat cannot account for.",
    "card_value": "What this face-up card is worth to the plan.",
    "needed_bonus": "A colour the plan is short of: a known card beats a blind one.",
    "expected_settlement": "Ticket points expected at the end: plus their value if "
    "completed, minus it if not, weighted by the odds the trains left can finish the plan.",
    "dead_tickets": "Tickets kept that can no longer be completed.",
    "added_track": "H2: train cars this keep adds to the plan; it keeps as little as it can.",
    "tickets_kept": "Fewer tickets kept ranks higher.",
    "priority": "H1's rule: a claim, then tickets, then a face-up locomotive, then the "
    "face-up colour it holds most of, then the deck.",
    "cards_held": "H1: cards of this colour already in hand.",
    "route_cars": "H1: train cars the kept tickets' routes need.",
}

#: Inputs behind the terms, each in its own unit.
FACTS = {
    "cars": "The route's length, in train cars.",
    "on_plan": "1 if the segment is on the planned network for the held tickets.",
    "detour": "Extra cars the plan would need if this segment were lost.",
    "needed_cards_spent": "Cards spent on this gray route of a colour the plan needs.",
    "locomotives": "Locomotives this payment uses.",
    "pending_detour": "H4: the largest detour among planned segments it cannot pay for yet.",
    "unseen_cards": "Cards this seat cannot account for: the deck and opponents' blind draws.",
    "plan_needs": "Cards of this colour the plan still needs beyond those in hand.",
    "plan_cost_after": "Train cars the plan would need after this.",
    "completion_odds": "The odds the trains left can finish the plan: even at the target "
    "share of trains, falling as the plan outgrows them.",
    "mean_ticket_points": "The average value of the tickets still in the deck.",
    "ticket_points": "Points riding on the tickets this keep holds.",
    "added_cars": "Train cars this keep adds to the plan.",
}

#: Quantities of the whole decision.
CONTEXT = {
    "plan_cost": "Train cars still needed to connect every held ticket.",
    "plan_points": "Points riding on the held tickets still open and reachable.",
    "plan_segments": "Unclaimed segments on the planned network.",
    "unreachable_tickets": "Held tickets no route can complete any more.",
    "plan_car_value": "What a car of progress on the plan is worth, in points: more when "
    "the plan is nearly done.",
    "trains": "Trains this seat has left.",
    "endgame": "1 once trains run low and play turns to finishing what can be finished.",
    "plan_completable": "1 if the trains left can still finish the plan.",
    "progress": "How far through the game it is, by the fastest seat's trains spent.",
    "final_turn": "1 once the final round has started: this is the seat's last turn, so a "
    "claim is worth only the points it scores now.",
}

#: Rules that rank a candidate.
RULES = {
    "forced": "The only legal move.",
    "uniform": "H0 picks uniformly at random.",
    "claim_most_points": "H1: claim the affordable route worth the most points.",
    "draw_tickets": "H1: draw tickets while it holds few and has trains to spare.",
    "tickets_unwanted": "H1: enough tickets, or too few trains, to draw more.",
    "faceup_locomotive": "H1: take a face-up locomotive.",
    "faceup_most_held": "H1: take the face-up colour it holds most of.",
    "blind_draw": "H1: draw from the deck rather than start a colour it holds none of.",
    "faceup_unheld": "H1: a face-up colour it holds none of.",
    "pass": "No other move.",
    "keep_fewest_shortest": "H1: keep the fewest tickets allowed, the shortest routes.",
    "ticket_gate": "Too few trains left to take on new tickets.",
    "final_turn": "Its last turn: a card or ticket drawn now can never be used.",
}


def glossary() -> p.Glossary:
    return p.Glossary(terms=TERMS, facts=FACTS, context=CONTEXT, rules=RULES)
