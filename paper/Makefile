# Makefile for compiling the IEEEtran paper

TARGET = main
TEX = pdflatex
BIB = bibtex

all: $(TARGET).pdf

$(TARGET).pdf: $(TARGET).tex references.bib IEEEtran.cls
	$(TEX) $(TARGET)
	$(BIB) $(TARGET)
	$(TEX) $(TARGET)
	$(TEX) $(TARGET)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk

distclean: clean
	rm -f $(TARGET).pdf

.PHONY: all clean distclean
