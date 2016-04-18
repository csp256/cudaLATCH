#!/usr/bin/perl -w
use strict;

sub usage {
	print "Usage: $0 <options>\n";
	print <<OEF;
where options are (in order):

  NumberOfStreams                        How many streams to plot (M)

  NumberOfWindows                        How many windows to use (N)

  Window0_WindowSampleSize               this many samples per window for window0
  <Window1_WindowSampleSize>             ...for window1
...
  <WindowN_WindowSampleSize>             ...for windowN

  Window0_YRangeMin Window0_YRangeMax    Min and Max values for window0
  <Window1_YRangeMin Window1_YRangeMax>  ...for window1
...
  <WindowN_YRangeMin WindowN_YRangeMax>  ...for windowN

  Window0_geometry			 WIDTHxHEIGHT+XOFF+YOFF (in pixels)
  <Window1_geometry>                     ...for window1
...
  <WindowN_geometry>                     ...for windowN

  Stream0_Title                          Title used for stream 0
  <Stream1_Title>                        ...for stream1
...
  <StreamN_Title>                        ...for streamM

  WindowNumber0                          Window into which stream 0 is plotted
  <WindowNumber1>                        ... for stream1
...
  <WindowNumberM>                        ... for streamM
OEF
	exit(1);
}

sub WrongParameter {
	my $cause = shift;
	print "Expected parameter missing ($cause)...\n\n";
	usage;
	exit(1);
}


sub main {
	my $argIdx = 0;
	my $numberOfStreams = shift or WrongParameter("number of streams");
    my $numberOfWindows = shift or WrongParameter("number of windows");
	print "Will display $numberOfStreams Streams in $numberOfWindows windows...\n";
	my @sampleSizes;
	for(my $i=0; $i<$numberOfWindows; $i++) {
		my $samples = shift or WrongParameter("sample size $i");
		push @sampleSizes, $samples;
		print "Window $i will use a window of $samples samples\n";
	}
	my @ranges;
	for(my $i=0; $i<$numberOfWindows; $i++) {
		my $miny = shift;
		WrongParameter("min y of window $i") if !defined($miny);
		my $maxy = shift;
		WrongParameter("max y of window $i") if !defined($maxy);
		push @ranges, [ $miny, $maxy ];
		print "Window $i will use a range of [$miny, $maxy]\n";
	}
	my @geometries;
	for(my $i=0; $i<$numberOfWindows; $i++) {
		my $geometry = shift or WrongParameter("geometry $i");
		push @geometries, $geometry;
		print "Window $i will use a geometry of '$geometry'\n";
	}
	my @titles;
	for(my $i=0; $i<$numberOfStreams; $i++) {
		my $title = shift or WrongParameter("title $i");
		push @titles, $title;
		print "Stream $i will use a title of '$title'\n";
	}
    my @streams;    # streams in a window
    my @windows;    # window of a stream
    for(my $i=0; $i<$numberOfStreams; $i++) {
        my $window = shift;
		WrongParameter("window of stream $i") if !defined $window;
        push @{$streams[$window]}, $i;
        $windows[$i] = $window;
        print "Stream $i will be plotted in window $window\n";
    }
    # check that every window has a stream
	for my $windowIdx(0..$numberOfWindows-1) {
		if (!defined($streams[$windowIdx]) or @{$streams[$windowIdx]} == 0) {
			warn "Warning: Window $windowIdx has no streams!\n";
		}
	}
	my @gnuplots;
	my @buffers;
	my @xcounters;
    for (0..$numberOfStreams-1) {
		my @data = [];
		push @buffers, @data;
		push @xcounters, 0;
    }
	for(my $i=0; $i<$numberOfWindows; $i++) {
		local *PIPE;
		my $geometry = $geometries[$i];
		open PIPE, "|gnuplot -geometry $geometry" || die "Can't initialize gnuplot number ".($i+1)."\n";
		select((select(PIPE), $| = 1)[0]);
		push @gnuplots, *PIPE;
		print PIPE "set xtics\n";
		print PIPE "set ytics\n";
#		print PIPE "set yrange [".($ranges[$i]->[0]).":".($ranges[$i]->[1])."]\n";
		print PIPE "set style data lines\n";
		print PIPE "set grid\n";
		print PIPE "set term x11\n";
	}
	my $streamIdx = 0;
	# replace @ARGV with remaining args for <> below
	@ARGV = @_;
	while(<>) {
		chomp;
		my @parts = split /:/;
		#print "$.: parts=", join("-", @parts), "\n";
		$streamIdx = $parts[0];
        my $windowIdx = $windows[$streamIdx];
        my $buf = $buffers[$streamIdx];
		my $pip = $gnuplots[$windowIdx];
		# data buffering (up to stream sample size)
		my $xcounter = $xcounters[$streamIdx];
		push @{$buf}, "$xcounter $parts[1]";
		$xcounters[$streamIdx]++;
        my $max_xcounter = $xcounter;
		my $q = 0;
        for my $stream (@{$streams[$windowIdx]}) {
            if ($xcounters[$stream] > $max_xcounter) {
                $max_xcounter = $xcounters[$stream];
				$q = 1;
            }
        }
		my $plotInterval = 15;
		if ($max_xcounter % $plotInterval != $plotInterval-1 && $q == 1) {
			next;
		}

		print $pip "set xrange [".($max_xcounter-$sampleSizes[$windowIdx]).":".($max_xcounter)."]\n";
        my @plots;
        for my $stream (@{$streams[$windowIdx]}) {
			if (@{$buffers[$stream]} > 0) {
				push @plots, "\"-\" title '$titles[$stream]'";
			}
        }
		print $pip "plot ", join(", ", @plots), "\n";
        for my $stream (@{$streams[$windowIdx]}) {
			if (@{$buffers[$stream]} > 0) {
				for my $elem (reverse @{$buffers[$stream]}) {
					print $pip "$elem\n";
				}
				print $pip "e\n";
			}
        }
		if (scalar(@{$buf})>$sampleSizes[$windowIdx]) {
			shift @{$buf};
		}
	}
	for(my $i=0; $i<$numberOfWindows; $i++) {
		my $pip = $gnuplots[$i];
		print $pip "exit;\n";
		close $pip;
	}
}

main @ARGV;
