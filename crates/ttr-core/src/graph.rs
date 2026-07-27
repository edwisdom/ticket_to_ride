//! Graph work the engine needs. Everything here is a pure function of its arguments.
//!
//! No caching: callers that want it own it, because they know when it goes stale. The
//! union-find is the only piece `step()` touches; the rest is used by scoring and the
//! observation encoder.

/// Path *halving*: one pointer update per step, no second pass, no recursion.
///
/// A clone-based state can compress freely; an undo-log design could not, which is one of
/// the reasons the state is cloned rather than journalled (PLAN.md §5.1).
///
/// Note that halving does **not** flatten a chain in one pass -- a test asserting
/// `parent[x] == root` after a single find on a long chain will fail. Repeated finds
/// converge, and that is the trade being made.
#[inline]
pub fn dsu_find(parent: &mut [u8], mut x: u8) -> u8 {
    while parent[x as usize] != x {
        let grandparent = parent[parent[x as usize] as usize];
        parent[x as usize] = grandparent;
        x = grandparent;
    }
    x
}

/// Union by **smaller index**, so the forest is a deterministic function of the claims
/// rather than of the order they happened to be applied in.
pub fn dsu_union(parent: &mut [u8], a: u8, b: u8) -> bool {
    let (mut ra, mut rb) = (dsu_find(parent, a), dsu_find(parent, b));
    if ra == rb {
        return false;
    }
    if ra > rb {
        std::mem::swap(&mut ra, &mut rb);
    }
    parent[rb as usize] = ra;
    true
}

#[inline]
pub fn dsu_connected(parent: &mut [u8], a: u8, b: u8) -> bool {
    dsu_find(parent, a) == dsu_find(parent, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fresh(n: u8) -> Vec<u8> {
        (0..n).collect()
    }

    #[test]
    fn union_and_find_agree() {
        let mut p = fresh(10);
        assert!(!dsu_connected(&mut p, 0, 5));
        assert!(dsu_union(&mut p, 0, 5));
        assert!(dsu_connected(&mut p, 0, 5));
        assert!(
            !dsu_union(&mut p, 5, 0),
            "a redundant union must report false"
        );
    }

    #[test]
    fn the_root_is_the_smallest_index_in_the_component() {
        let mut p = fresh(10);
        for (a, b) in [(7, 3), (3, 9), (9, 1)] {
            dsu_union(&mut p, a, b);
        }
        for v in [1, 3, 7, 9] {
            assert_eq!(dsu_find(&mut p, v), 1);
        }
    }

    #[test]
    fn path_halving_does_not_flatten_in_one_pass() {
        // Documented, not a defect: halving updates one pointer per step. A test that
        // asserts full flattening after a single find is testing path *compression*, a
        // different algorithm, and would fail here for the right reason.
        let mut p: Vec<u8> = (0..20).collect();
        for i in (1..20).rev() {
            p[i] = (i - 1) as u8;
        }
        assert_eq!(dsu_find(&mut p, 19), 0);
        assert_ne!(p[19], 0, "one halving pass should not reach the root");
        for _ in 0..10 {
            dsu_find(&mut p, 19);
        }
        assert_eq!(p[19], 0, "repeated finds must converge");
    }
}
