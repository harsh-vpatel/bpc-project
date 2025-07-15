#!/usr/bin/env perl

use utf8;
use Encode;

# Simple tokenizer that splits on whitespace and preserves sentence boundaries
# Each input line is treated as one sentence

my $first_line = 1;
while (<>) {
    my $output = "";
    $_ = decode('utf8',$_);

    # delete optional byte order markers (BOM)
    if ($first_line) {
        undef $first_line;
        s/^\x{FEFF}//;
    }

    # Remove leading/trailing whitespace
    s/^\s+//;
    s/\s+$//;
    
    # Skip empty lines
    next if /^\s*$/;
    
    # Split on whitespace
    my @words = split(/\s+/);
    
    # Process each word to separate punctuation
    for my $word (@words) {
        # Handle leading punctuation
        while ($word =~ s/^([¿¡{\'\\`"‚„†‡‹''""•–—›»«\(\[\"])(.+)/$2/) {
            $output .= "$1\n";
        }
        
        # Handle trailing punctuation
        my @trailing = ();
        while ($word =~ s/(.+)([}\'\`\",;:\!\?\%‚„…†‡‰‹''""•–—›»«\)\]\.])\s*$/$1/) {
            unshift @trailing, $2;  # Add to front to preserve order
        }
        
        # Output the main word
        $output .= "$word\n" if $word ne '';
        
        # Output trailing punctuation
        for my $punct (@trailing) {
            $output .= "$punct\n";
        }
    }
    
    # Add sentence boundary (empty line) after each input line
    $output .= "\n";
    
    print encode('utf-8', $output);
}
