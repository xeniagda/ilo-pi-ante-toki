use std::collections::{HashMap, HashSet};
use std::hash::Hash;
use std::fmt::Debug;

use std::io::Write;

#[derive(Debug, Copy, Clone)]
pub enum Gram<I> {
    Orig(I),
    Composition(usize, usize),
}

// Retuns the tokenized text, along with a list of decompositions
pub fn encode_into_ngrams<I: Debug + Copy + Eq + Hash, F: Fn(&I) -> bool>(inp: Vec<I>, rel_lim: f64, can_pair: F) -> (Vec<usize>, Vec<Gram<I>>) {
    // Convert the text into orig tokens

    let inp_len = inp.len() as f64;

    let mut tokens: Vec<usize> = Vec::new();
    let mut grams = Vec::new();

    let mut skips = HashSet::new();

    let mut i2tok: HashMap<I, usize> = HashMap::new();
    for i in inp {
        if !i2tok.contains_key(&i) {
            let idx = grams.len();
            i2tok.insert(i, idx);
            grams.push(Gram::Orig(i));

            if !can_pair(&i) {
                skips.insert(idx);
            }
        }

        tokens.push(*i2tok.get(&i).unwrap());
    }

    // println!("Tokens: {:?}", tokens);
    // println!("Grams: {:?}", grams);

    loop {
        let pair_freq = get_pair_freq(&tokens, &skips);
        let (&commonest_pair, &freq) = if let Some(x) = pair_freq.iter().max_by_key(|(_, &count)| count) {
            x
        } else {
            eprintln!("Ran out of pairs: {:?} / {:?} - {:?}", tokens, grams, skips);
            return (tokens, grams);
        };
        let rel_freq = freq as f64 / inp_len as f64;
        println!("Commonest: {:?}, freq {}, {}%", commonest_pair, freq, rel_freq);
        if rel_freq < rel_lim {
            break;
        }

        let new_gram_idx = grams.len();
        grams.push(Gram::Composition(commonest_pair.0, commonest_pair.1));

        let mut contracted_tokens = Vec::with_capacity(tokens.len() - freq);
        let mut at = 0;
        while at < tokens.len() {
            let here = tokens[at];
            let next = tokens.get(at + 1);

            if here == commonest_pair.0 && next == Some(&commonest_pair.1) {
                contracted_tokens.push(new_gram_idx);
                at += 2;
            } else {
                contracted_tokens.push(here);
                at += 1;
            }
        }

        tokens = contracted_tokens;

        // println!("Tokens: {:?}", tokens);
        // println!("Grams: {:?}", grams);
    }

    (tokens, grams)
}

fn get_pair_freq(input: &[usize], skips: &HashSet<usize>) -> HashMap<(usize, usize), usize> {
    let mut counter = HashMap::new();

    for (&a, &b) in input.iter().zip(input.iter().skip(1)) {
        if skips.contains(&a) || skips.contains(&b) {
            continue; // Don't combine over word boundaries
        }

        let count = counter.entry((a, b)).or_insert(0);
        *count += 1;
    }

    counter
}

fn decompose_sequence<I: Eq + Hash + Copy>(mut tokens: Vec<usize>, grams: &[Gram<I>]) -> Vec<I> {
    // Make all tokens point into Orig-ngrams
    while {
        let mut new_tokens = Vec::new();
        let mut all_origs = true;

        for &tok in &tokens {
            match grams[tok] {
                Gram::Orig(_) => {
                    new_tokens.push(tok);
                }
                Gram::Composition(a, b) => {
                    new_tokens.extend_from_slice(&[a, b]);
                    all_origs = false;
                }
            }
        }

        tokens = new_tokens;

        !all_origs
    } {}

    // Now we know that grams[tok] == Gram::Orig(_) for all tok in tokens
    tokens
        .into_iter()
        .map(|g_idx| {
            match grams[g_idx] {
                Gram::Orig(ch) => ch,
                _ => unreachable!(),
            }
        })
        .collect()
}

// Assumes ngrams are in topolocical order
// Format:
// Gram::Orig(ch): chl = utf-8 len of ch
// offset: 0x0 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x8
//  value: chl ---  ch...  --- 0x0 0x0 0x0 0x0
//
// Gram::Composition(a, b) encodes:
// offset: 0x0 0x1 0x2 0x3 0x4 0x5 0x6 0x7 0x8
//  value: 0x0 -------a------- -------b-------

pub fn encode_grams<F: Write>(out: &mut F, grams: Vec<Gram<char>>) -> std::io::Result<()> {
    // Assert topoorder
    for (i, gram) in grams.iter().enumerate() {
        match gram {
            &Gram::Composition(a, b) => debug_assert!(a < i && b < i),
            _ => {}
        }
    }

    for gram in grams {
        match gram {
            Gram::Orig(ch) => {
                let chl = ch.len_utf8();
                out.write(&[chl as u8])?; // Safe to covert since ch.len_utf8() <= 4
                let mut ch_buf = [0u8; 8];
                ch.encode_utf8(&mut ch_buf);
                out.write(&ch_buf)?;
            }
            Gram::Composition(a, b) => {
                out.write(&[0])?;

                let a_bytes = (a as u32).to_le_bytes();
                let b_bytes = (b as u32).to_le_bytes();

                out.write(&a_bytes)?;
                out.write(&b_bytes)?;
            }
        }
    }

    Ok(())
}

#[allow(unused)]
fn main() -> std::io::Result<()> {
    use std::io::Read;

    let mut f = std::fs::File::open("/Users/jonathanloov/.cache/ilo-pi-ante-toki/sentences-sec.txt")?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;

    let inp = String::from_utf8_lossy(&buf).chars().collect();

    println!("Encoding");
    let (stream, grams) = encode_into_ngrams(inp, 0.001, |&_ch| true);

    // println!("{:?}", decompose_sequence(stream, &grams));
    for i in 0..grams.len() {
        if !stream.contains(&i) {
            print!("  ");
        } else {
            print!("* ");
        }
        let expanded_gram = decompose_sequence(vec![i], &grams);
        println!("{} -> {:?}", i, expanded_gram.iter().collect::<String>());
    }

    let mut out = std::fs::File::create("/tmp/gramcode.bin")?;
    encode_grams(&mut out, grams)?;

    Ok(())
}
