// /**https://en.wikipedia.org/wiki/Needleman%E2%80%93Wunsch_algorithm
// Returns matrix of shape (length_b,length_a)*/
// pub fn needleman_wunsch<T, A, B, IA: Iterator<Item=A>, IB: Iterator<Item=B>>(a: impl Fn() -> IA, length_a: usize,
//                                                                              b: impl Fn() -> IB, length_b: usize,
//                                                                              cost: impl Fn(Option<A>, Option<B>) -> T,
//                                                                              zero: T,
//                                                                              max: impl Fn(&T, &T) -> T) -> Vec<T> {
//     let mut table = Vec::with_capacity(length_b*length_a);
//     ta
//     for (index_b, code_b) in b().enumerate() {
//         result = index_b;
//         let mut distance_a = index_b;
//
//         for (index_a, code_a) in a().enumerate() {
//             distance_b = if code_a == code_b {
//                 distance_a
//             } else {
//                 distance_a + 1
//             };
//
//             distance_a = cache[index_a];
//
//             result = if distance_a > result {
//                 if distance_b > result {
//                     result + 1
//                 } else {
//                     distance_b
//                 }
//             } else if distance_b > distance_a {
//                 distance_a + 1
//             } else {
//                 distance_b
//             };
//
//             cache[index_a] = result;
//         }
//     }
//     table
// }
//
// /**
// https://en.m.wikipedia.org/wiki/Hirschberg%27s_algorithm
// For best performance make sure that a_length <= b_length
//
//  */
// #[must_use]
// pub fn hirschberg<T, IA: Iterator<Item=T>, IB: Iterator<Item=T>>(a: impl Fn() -> IA, length_a: usize, b: impl Fn() -> IB) -> usize {
//     let mut result = 0;
//     /* Initialize the vector.
//      *
//      * This is why it’s fast, normally a matrix is used,
//      * here we use a single vector. */
//     let mut cache: Vec<usize> = (1..).take(length_a).collect();
//     let mut distance_b;
//
//     /* Loop. */
//     for (index_b, code_b) in b().enumerate() {
//         result = index_b;
//         let mut distance_a = index_b;
//
//         for (index_a, code_a) in a().enumerate() {
//             distance_b = if code_a == code_b {
//                 distance_a
//             } else {
//                 distance_a + 1
//             };
//
//             distance_a = cache[index_a];
//
//             result = if distance_a > result {
//                 if distance_b > result {
//                     result + 1
//                 } else {
//                     distance_b
//                 }
//             } else if distance_b > distance_a {
//                 distance_a + 1
//             } else {
//                 distance_b
//             };
//
//             cache[index_a] = result;
//         }
//     }
//
//     result
// }
//
//
// #[must_use]
// pub fn levenshtein(a: &str, b: &str) -> usize {
//     /* Shortcut optimizations / degenerate cases. */
//     if a == b {
//         return 0;
//     }
//
//     let length_a = a.chars().count();
//     let length_b = b.chars().count();
//
//     if length_a == 0 {
//         return length_b;
//     }
//
//     if length_b == 0 {
//         return length_a;
//     }
//     if length_b < length_a {
//         hirschberg(|| b.chars(), length_b, || a.chars())
//     } else {
//         hirschberg(|| a.chars(), length_a, || b.chars())
//     }
// }
//
// #[cfg(test)]
// mod tests {
//     use super::*;
//
//     #[test]
//     fn empty_left() {
//         assert_eq!(levenshtein("", "a"), 1);
//     }
//
//     #[test]
//     fn empty_right() {
//         assert_eq!(levenshtein("a", ""), 1);
//     }
//
//     #[test]
//     fn empty_both() {
//         assert_eq!(levenshtein("", ""), 0);
//     }
//
//     #[test]
//     fn equal_long() {
//         assert_eq!(levenshtein("levenshtein", "levenshtein"), 0);
//     }
//
//     #[test]
//     fn case_sensitive() {
//         assert_eq!(levenshtein("DwAyNE", "DUANE"), 2);
//         assert_eq!(levenshtein("dwayne", "DuAnE"), 5);
//     }
//
//     #[test]
//     fn ordering() {
//         assert_eq!(levenshtein("aarrgh", "aargh"), 1);
//         assert_eq!(levenshtein("aargh", "aarrgh"), 1);
//     }
//
//     #[test]
//     fn should_work() {
//         assert_eq!(levenshtein("sitting", "kitten"), 3);
//         assert_eq!(levenshtein("gumbo", "gambol"), 2);
//         assert_eq!(levenshtein("saturday", "sunday"), 3);
//         assert_eq!(levenshtein("a", "b"), 1);
//         assert_eq!(levenshtein("ab", "ac"), 1);
//         assert_eq!(levenshtein("ac", "bc"), 1);
//         assert_eq!(levenshtein("abc", "axc"), 1);
//         assert_eq!(levenshtein("xabxcdxxefxgx", "1ab2cd34ef5g6"), 6);
//         assert_eq!(levenshtein("xabxcdxxefxgx", "abcdefg"), 6);
//         assert_eq!(levenshtein("javawasneat", "scalaisgreat"), 7);
//         assert_eq!(levenshtein("example", "samples"), 3);
//         assert_eq!(levenshtein("sturgeon", "urgently"), 6);
//         assert_eq!(levenshtein("levenshtein", "frankenstein"), 6);
//         assert_eq!(levenshtein("distance", "difference"), 5);
//         assert_eq!(levenshtein("kitten", "sitting"), 3);
//         assert_eq!(levenshtein("Tier", "Tor"), 2);
//     }
//
//     #[test]
//     fn unicode() {
//         assert_eq!(levenshtein("😄", "😦"), 1);
//         assert_eq!(levenshtein("😘", "😘"), 0);
//     }
// }