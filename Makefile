# Filename
SRC = presentation.md
THEME := "Frankfurt"

PDFS=$(SRC:.md=.pdf)
HTML=$(SRC:.md=.html)

all:	$(PDFS) $(HTML)

pdf:	clean $(PDFS)
html:	clean $(HTML)

%.html:	%.md
	pandoc -t revealjs -s  -o $@ $< &

#%.pdf:	%.md
#	pandoc -t beamer -s -V theme=$(THEME) --latex-engine=pdflatex -o $@ $< &

clean:
	rm -f *.html *.pdf
