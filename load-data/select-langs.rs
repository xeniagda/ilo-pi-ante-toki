// This program removes all sentences which does not either have primary-secondary pair, or primary-auxiliary pair
// It then separates each language's sentences into separate files, and creates a two links files with information about
// where each sentence starts and ends in each language.

// The generated sentence files contains every sentence back to back, with no separators.
// The links files contains links between the sentences. Each sentence link is encoded as
// four numbers. The four numbers describe the start of the primary sentence, the end of it, the start of the
// secondary/auxiliary sentence and the end of it.
// In the binary format, each number is encoded as a 32-bit unsigned integer

mod tokens;

use std::io::{Write, Result, BufWriter, BufReader, BufRead, Error, ErrorKind};
use std::fs::File;
use std::env::var;
use std::collections::{HashMap, HashSet};
use std::convert::TryInto;

const PRIM_LANGUAGE: &str = "eng";
const SEC_LANGUAGE: &str = "toki";
const AUX_LANGUAGE: &str = "spa";

const REL_LIM: f64 = 0.005;

// Described above
// TODO: Either make this a compile-time flag, or a CLI-argument
const BINARY_MODE: bool = true;

fn write_binary_number_to_file<F: Write>(file: &mut F, number: u32) -> Result<()> {
    let buf = number.to_le_bytes();
    file.write(&buf)?;

    Ok(())
}

fn write_ascii_number_to_file<F: Write>(file: &mut F, number: u32) -> Result<()> {
    let txt = format!("{} ", number);
    file.write(txt.as_bytes())?;

    Ok(())
}

fn write_number_to_file<F: Write>(file: &mut F, number: u32) -> Result<()> {
    if BINARY_MODE {
        write_binary_number_to_file(file, number)
    } else {
        write_ascii_number_to_file(file, number)
    }
}

#[derive(Debug)]
struct Translation<SentenceContent> {
    prim_language: HashMap<u32, SentenceContent>,
    sec_language: HashMap<u32, SentenceContent>,
    aux_language: HashMap<u32, SentenceContent>,
    links: HashSet<(u32, u32)>, // Always (prim, sec) or (prim, aux)
}

impl <SentenceContent> Translation<SentenceContent> {
    fn new() -> Self {
        Translation {
            prim_language: HashMap::new(),
            sec_language: HashMap::new(),
            aux_language: HashMap::new(),
            links: HashSet::new(),
        }
    }
}

impl Translation<Vec<u8>> {
    fn consume_sentences<F: BufRead>(&mut self, mut file: F) -> Result<usize> {
        let mut counter = 0;

        loop {
            let mut id_buf = Vec::new();
            file.read_until('\t' as u8, &mut id_buf)?;

            // Remove tab, quit if not found
            if id_buf.pop() != Some('\t' as u8) {
                break;
            }

            let mut language_buf = Vec::new();
            file.read_until('\t' as u8, &mut language_buf)?;
            language_buf.pop(); // Remove tab
            let language = String::from_utf8_lossy(&language_buf);

            let mut sentence = Vec::new();
            file.read_until('\n' as u8, &mut sentence)?;
            sentence.pop(); // Remove newline

            if language != PRIM_LANGUAGE && language != SEC_LANGUAGE && language != AUX_LANGUAGE {
                // println!("Skip");
                continue;
            }

            let id_st = String::from_utf8(id_buf).unwrap();
            let id_n: u32 = id_st.parse().unwrap();

            // println!("ID: {:?}, Language: {:?}, Sentence: {:?}", id_n, language, sentence);

            let list_to_add =
                match &*language {
                    PRIM_LANGUAGE => &mut self.prim_language,
                    SEC_LANGUAGE => &mut self.sec_language,
                    AUX_LANGUAGE => &mut self.aux_language,
                    _ => unreachable!() // We checked before that language belonged to one of the previous alternatives
                };

            list_to_add.insert(id_n, sentence.to_owned());
            counter += 1;
        }

        Ok(counter)
    }

    fn consume_links<F: BufRead>(&mut self, mut file: F, remove_unlinked: bool) -> Result<(usize, usize)> {
        let mut n_read = 0;
        let mut n_wrong = 0;

        let mut prim_ids = HashSet::new();
        let mut sec_ids = HashSet::new();
        let mut aux_ids = HashSet::new();

        loop {
            let mut first_buf = Vec::new();
            file.read_until('\t' as u8, &mut first_buf)?;

            // Remove tab, quit if not found
            if first_buf.pop() != Some('\t' as u8) {
                break;
            }

            let mut second_buf = Vec::new();
            file.read_until('\n' as u8, &mut second_buf)?;
            second_buf.pop();


            let first_st = String::from_utf8(first_buf).unwrap();
            let first_n: u32 = first_st.parse().unwrap();

            let second_st = String::from_utf8(second_buf).unwrap();
            let second_n: u32 = second_st.parse().unwrap();

            match (self.prim_language.contains_key(&first_n), self.sec_language.contains_key(&second_n), self.aux_language.contains_key(&second_n)) {
                (true, true, false) => {
                    self.links.insert((first_n, second_n));
                    prim_ids.insert(first_n);
                    sec_ids.insert(second_n);
                    n_read += 1;
                    continue;
                }
                (true, false, true) => {
                    self.links.insert((first_n, second_n));
                    prim_ids.insert(first_n);
                    aux_ids.insert(second_n);
                    n_read += 1;
                    continue;
                }
                _ => {}
            }
            match (self.prim_language.contains_key(&second_n), self.sec_language.contains_key(&first_n), self.aux_language.contains_key(&first_n)) {
                (true, true, false) => {
                    self.links.insert((second_n, first_n));
                    prim_ids.insert(second_n);
                    sec_ids.insert(first_n);
                    n_read += 1;
                    continue;
                }
                (true, false, true) => {
                    self.links.insert((second_n, first_n));
                    prim_ids.insert(second_n);
                    aux_ids.insert(first_n);
                    n_read += 1;
                    continue;
                }
                _ => {}
            }
            n_wrong += 1;
        }

        if remove_unlinked {
            println!("Keeping {:?}/{:?}/{:?}", prim_ids.len(), sec_ids.len(), aux_ids.len());
            self.prim_language.retain(|&id, _| prim_ids.contains(&id));
            self.sec_language.retain(|&id, _| sec_ids.contains(&id));
            self.aux_language.retain(|&id, _| aux_ids.contains(&id));
        }

        Ok((n_read, n_wrong))
    }

    fn stringify(self) -> Result<Translation<String>> {
        let prim_language =
            self.prim_language
            .into_iter()
            .map(|(k, v)| {
                let st = String::from_utf8(v).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid UTF-8!"))?;
                Ok((k, st))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let sec_language =
            self.sec_language
            .into_iter()
            .map(|(k, v)| {
                let st = String::from_utf8(v).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid UTF-8!"))?;
                Ok((k, st))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        let aux_language =
            self.aux_language
            .into_iter()
            .map(|(k, v)| {
                let st = String::from_utf8(v).map_err(|_| Error::new(ErrorKind::InvalidData, "Invalid UTF-8!"))?;
                Ok((k, st))
            })
            .collect::<Result<HashMap<_, _>>>()?;

        Ok(Translation {
            prim_language, sec_language, aux_language,
            links: self.links,
        })
    }
}

impl <T> Translation<T> {
    fn write_links<F: Write>(&self, file: &mut F, id_offset_size: &HashMap<u32, (usize, usize)>, write_secondary: bool) -> Result<()> {
        for &(prim_id, other_id) in &self.links {
            match (self.sec_language.contains_key(&other_id), write_secondary) {
                (true, false) | (false, true) => continue,
                _ => {}
            }

            let (prim_offset, prim_len) = id_offset_size.get(&prim_id).unwrap();
            let (other_offset, other_len) = id_offset_size.get(&other_id).unwrap();

            write_number_to_file(file, *prim_offset as u32)?;
            write_number_to_file(file, *prim_len as u32)?;
            write_number_to_file(file, *other_offset as u32)?;
            write_number_to_file(file, *other_len as u32)?;
        }
        Ok(())
    }
}

fn gramify_sentences(sents: HashMap<u32, String>) -> (HashMap<u32, Vec<usize>>, Gramophone) {
    let gram = Gramophone::from_word_iter(
        sents
            .values()
            .map(|x| x.chars())
    );
    let grammed_sents =
        sents
        .into_iter()
        .map(|(k, sent)| (k, gram.encode_text(sent.chars())))
        .collect::<HashMap<_, _>>();

    (grammed_sents, gram)
}

impl Translation<String> {
    fn gramify(self) -> (Translation<Vec<usize>>, Gramophone, Gramophone, Gramophone) {
        let (prim_language, prim_gram) = gramify_sentences(self.prim_language);
        let (sec_language, sec_gram) = gramify_sentences(self.sec_language);
        let (aux_language, aux_gram) = gramify_sentences(self.aux_language);

        let trans = Translation {
            prim_language, sec_language, aux_language,
            links: self.links,
        };

        (trans, prim_gram, sec_gram, aux_gram)
    }
}

impl Translation<Vec<usize>> {
    fn write_sentences<F: Write>(&self, file: &mut F, from: u8) -> Result<HashMap<u32, (usize, usize)>> {
        let mut id_offset_size: HashMap<u32, _> = HashMap::new();
        let mut offset = 0;

        let sentences = match from {
            0 => &self.prim_language,
            1 => &self.sec_language,
            2 => &self.aux_language,
            _ => unimplemented!(),
        };

        for (&id, sentence) in sentences {
            id_offset_size.insert(id, (offset, sentence.len()));


            for &point in sentence {
                let point_u16: u16 = point.try_into().map_err(|_| Error::new(ErrorKind::InvalidData, format!("{} is too large to fit in a u16!", point)))?;
                file.write(&point_u16.to_le_bytes())?;
            }
            offset += sentence.len() * 2;
        }

        Ok(id_offset_size)
    }
}

struct Gramophone {
    grams: Vec<tokens::Gram<char>>,
    i2idx: HashMap<char, usize>,
}

impl Gramophone {
    // Assumes no zeros in iter
    fn from_word_iter<
            I: IntoIterator<Item=J>,
            J: IntoIterator<Item=char>,
        >(iter: I) -> Gramophone {
        let mut inp = Vec::new();
        for word in iter {
            inp.extend(word);
            inp.push('\0');
        }

        let (_, grams) = tokens::encode_into_ngrams(inp, REL_LIM, |&x| x != '\0');

        let mut i2idx = HashMap::new();
        for (idx, gram) in grams.iter().enumerate() {
            if let &tokens::Gram::Orig(i) = gram {
                i2idx.insert(i, idx);
            }
        }

        Gramophone {
            grams,
            i2idx,
        }
    }

    fn encode_text<I: IntoIterator<Item=char>>(&self, text: I) -> Vec<usize> {
        let mut tokens = Vec::new();
        for ch in text {
            tokens.push(*self.i2idx.get(&ch).expect(&format!("no such char: {:?}", ch)));
        }

        for (i, gram) in self.grams.iter().enumerate() {
            let (a, b) = if let &tokens::Gram::Composition(a, b) = gram {
                (a, b)
            } else {
                continue;
            };

            let mut contracted_tokens = Vec::new();
            let mut at = 0;
            while at < tokens.len() {
                let here = tokens[at];
                let next = tokens.get(at + 1);

                if here == a && next == Some(&b) {
                    contracted_tokens.push(i);
                    at += 2;
                } else {
                    contracted_tokens.push(here);
                    at += 1;
                }
            }

            tokens = contracted_tokens;
        }

        tokens
    }
}

fn get_cache_path(filename: &str) -> String {
    format!("{}/.cache/ilo-pi-ante-toki/{}", var("HOME").expect("no home"), filename)
}

fn main() -> Result<()> {
    let sentence_file = BufReader::new(File::open(get_cache_path("raw/sentences.tsv"))?);
    let links_file = BufReader::new(File::open(get_cache_path("raw/links.tsv"))?);

    let mut sentences = Translation::new();

    println!("Consuming sentences");
    sentences.consume_sentences(sentence_file)?;
    println!("Loaded {:?}/{:?}/{:?}", sentences.prim_language.len(), sentences.sec_language.len(), sentences.aux_language.len());

    println!("Consuming links");
    let (read, wrong) = sentences.consume_links(links_file, true)?;
    println!("Loaded {:?} links ({:?} were wrong)", read, wrong);

    println!("After filter {:?}/{:?}/{:?}", sentences.prim_language.len(), sentences.sec_language.len(), sentences.aux_language.len());

    println!("Stringifying");
    let sent_string = sentences.stringify()?;
    println!("Gramifying");
    let (sent_ngram, prim_gram, sec_gram, aux_gram) = sent_string.gramify();

    println!("Writing primary ngrams");
    let mut prim_ngrams = BufWriter::new(File::create(get_cache_path("ngrams-prim.bin"))?);
    tokens::encode_grams(&mut prim_ngrams, prim_gram.grams)?;
    prim_ngrams.flush()?;

    println!("Writing secondary ngrams");
    let mut sec_ngrams = BufWriter::new(File::create(get_cache_path("ngrams-sec.bin"))?);
    tokens::encode_grams(&mut sec_ngrams, sec_gram.grams)?;
    sec_ngrams.flush()?;

    println!("Writing auxiliary ngrams");
    let mut aux_ngrams = BufWriter::new(File::create(get_cache_path("ngrams-aux.bin"))?);
    tokens::encode_grams(&mut aux_ngrams, aux_gram.grams)?;
    aux_ngrams.flush()?;

    println!("Writing primary sentences");
    let mut prim_output = BufWriter::new(File::create(get_cache_path("sentences-prim.bin"))?);
    let mut meta = sent_ngram.write_sentences(&mut prim_output, 0)?;
    prim_output.flush()?;

    println!("Writing secondary sentences");
    let mut sec_output = BufWriter::new(File::create(get_cache_path("sentences-sec.bin"))?);
    let sec_meta = sent_ngram.write_sentences(&mut sec_output, 1)?;
    sec_output.flush()?;

    meta.extend(sec_meta.into_iter());

    println!("Writing auxiliary sentences");
    let mut aux_output = BufWriter::new(File::create(get_cache_path("sentences-aux.bin"))?);
    let aux_meta = sent_ngram.write_sentences(&mut aux_output, 2)?;
    sec_output.flush()?;

    meta.extend(aux_meta.into_iter());

    println!("Writing secondary links");
    let mut links_output = BufWriter::new(File::create(get_cache_path("sec-links.bin"))?);
    sent_ngram.write_links(&mut links_output, &meta, true)?;
    links_output.flush()?;

    println!("Writing auxiliary links");
    let mut links_output = BufWriter::new(File::create(get_cache_path("aux-links.bin"))?);
    sent_ngram.write_links(&mut links_output, &meta, false)?;
    links_output.flush()?;

    println!("Done!");
    Ok(())
}
