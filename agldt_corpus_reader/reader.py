#!/usr/bin/env python

from nltk.corpus.reader.xmldocs import XMLCorpusReader
from lxml import etree
from agldt_corpus_reader.utils import Sentence, Word, Artificial


class AGLDTReader(XMLCorpusReader):
    def __init__(self, root, fileids):
        XMLCorpusReader.__init__(self, root, fileids)

    def xml(self, fileid=None):
        

        # Make sure we have exactly one file -- no concatenating XML.
        if fileid is None and len(self._fileids) == 1:
            fileid = self._fileids[0]
        if not isinstance(fileid, str):
            raise TypeError('Expected a single file identifier string')

        # Read the XML in using lxml.etree.
        x = etree.parse(self.abspath(fileid))

        return x

    def _is_artificial(self, t):
        try:
            t.attrib["artificial"]
            return True
        except KeyError:
            return False

    def _set_prop_if_there(self, el, p, repl=None):
        """
        Generic function that tries to set a proprety by accessing the appropriate XML attribute;
        if the attribute is not there, either None or a replacement string is returned.
        This is used for properties like "cite" or "cid" that might or might not be set for all files.
        It is also useful if there are missing values in the rest of the annotation (e.g. missing head attached to
        root with `repl='0'`)
        """

        s = el.attrib.get(p)
        if not s:
            s = repl
        return s

    def get_sentences_metadata(self, fileids=None):
        """
        Obtain the metadata stored in the attributes of the sentence element.
        Return a list that is in sync with the other sentence methods.
        E.g. you can process sentence nodes and metadata by zipping metadata and annotated sentences together
        Parameters
        ----------
        fileids
        Returns
        -------
        """
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]
        smetadata = []
        for f in fileids:
            x = self.xml(f)
            sents = x.xpath("//sentence")

            for s in sents:
                sid = self._set_prop_if_there(s, "id")
                docid = self._set_prop_if_there(s, "document_id")
                subdoc = self._set_prop_if_there(s, "subdoc")
                m = Sentence(sid, docid, subdoc)
                smetadata.append(m)
        return smetadata

    def _get_sent_tokens(self, sentence_el):
        toks = []
        words = sentence_el.xpath("word")
        for w in words:
            wid = self._set_prop_if_there(w, "id")
            form = self._set_prop_if_there(w, "form", repl='_')
            lemma = self._set_prop_if_there(w, "lemma", repl='_')
            postag = self._set_prop_if_there(w, "postag", repl='_')
            head = self._set_prop_if_there(w, "head", '0')
            relation = self._set_prop_if_there(w, "relation", repl='ERR')
            cite = self._set_prop_if_there(w, "cite")
            if self._is_artificial(w):
                art_type = self._set_prop_if_there(w, "artificial")
                t = Artificial(wid, form, lemma, postag, head, relation, cite, art_type)
            else:
                t = Word(wid, form, lemma, postag, head, relation, cite)
            toks.append(t)
        return toks

    def _get_sents_el(self, fileids=None):
        if fileids is None:
            fileids = self._fileids
        elif isinstance(fileids, str):
            fileids = [fileids]

        sents = []

        for f in fileids:
            x = self.xml(f)
            s = x.xpath("//sentence")
            sents.extend(s)

        return sents

    def annotated_sents(self, fileids=None):
        sents = self._get_sents_el(fileids)
        annotated_sents = []
        for s in sents:
            toks = self._get_sent_tokens(s)
            annotated_sents.append(toks)
        return annotated_sents

    def sents(self, fileids=None):
        sentels = self._get_sents_el(fileids)
        sents = []
        for s in sentels:
            words = s.xpath("word")
            sents.append([w.attrib["form"] for w in words])
        return sents

    def annotated_words(self, fileids=None):
        asents = self.annotated_sents(fileids)
        return [w for s in asents for w in s]

    def words(self, fileids=None):
        awords = self.annotated_words(fileids)
        return [w.form for w in awords]

    def _is_governed_by_artificial(self, t, tokens):
        h = t.head
        for tok in tokens:
            if tok.id == h:
                if isinstance(tok, Artificial):
                    return True
                else:
                    return False

    def _find_true_head(self, t, tokens):
        """
        Checks a node's head. If this head is an Arificial then it (recursively) searches for the first
        non-artificial node that is at the root of the subtree.
        Otherwise, it simply returns the original node's head
        Parameters
        ----------
        t : namedtuple
            Word or Artificial node
        tokens : list
            the full sentence, as a list of Artificial or Word
        Returns
        -------
        str : the id of the first Word element governing the whole structure
        """
        arts_ids = [tok.id for tok in tokens if isinstance(tok, Artificial)]
        h = t.head
        if h not in arts_ids:
            return h
        else:
            for tok in tokens:
                if h == tok.id:
                    # then we found the target immediate head
                    true_head = tok.head
                    # now let's check if th is an artificial
                    if true_head in arts_ids:
                        true_head = self._find_true_head(tok, tokens)
            return true_head

    def sent_to_dggraph(self, sent, rootrel='PRED'):
        """
        Creates a Dependency Graph object from an AGLDT sentence
        Parameters
        ----------
        sent : list(named tuple)
            the AGLDT sentence
        Returns
        -------
        nltk.parse.DependencyGraph
        """

        from nltk.parse import DependencyGraph

        strsent = "\n".join(["{}\t{}\t{}\t{}".format(w.form, w.postag, w.head, w.relation) for w in sent])
        for w in sent:
            if w.head == '0':
                rootrel = w.relation
                break
        g = DependencyGraph(strsent, cell_separator="\t", top_relation_label=rootrel)

        return g

    def triples(self, annotated_sent, rootrel='PRED'):
        dg = self.sent_to_dggraph(annotated_sent, rootrel)
        r = dg.nodes[0]
        return dg.triples(r)

    def export_to_conll(self, annotated_sents, out_file, dialect='2009'):
        """
        # TODO: at the moment, it works with conll 2009 only
        Save the sentences passed to a CoNLL file.
        Parameters
        ----------
        annotated_sents : list
            list of annotated sentences. Each sentence must contain the token as named tuples:
            Word or Artificial. You can use the method `annotated_sentence` to get them
        out_file : str
            filename (and path) to save the output
        dialect : str
            the CoNLL dialect
        """
        import logging

        c = ""
        if dialect == '2009':
            #  ID FORM LEMMA PLEMMA POS PPOS FEAT PFEAT HEAD PHEAD DEPREL PDEPREL FILLPRED PRED APREDs
            l = "{}\t{}\t{}\t_\t{}\t_\t{}\t_\t{}\t_\t{}\t_\t_\t_\t_\n"

        for s in annotated_sents:
            toks = [t for t in s if isinstance(t, Word)]
            for w in toks:
                try:
                    realh = self._find_true_head(w,s)
                except RecursionError:
                    logging.error("Problem with the head of token: {}:{}".format(w.cite, w.id))
                    continue
                relation = w.relation if realh == w.head else "ExD"
                try:
                    pos = w.postag[0]
                except IndexError:
                    pos = "x--------"
                lemma = w.lemma if w.lemma is not None else "Unknown"
                feat = "|".join(w.postag)
                form = w.form.replace("̓", "'")
                c += l.format(w.id, w.form, lemma, pos, feat, realh, relation)
            c+="\n"

        with open(out_file, "w") as out:
            out.write(c)
