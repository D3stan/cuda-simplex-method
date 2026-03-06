void Leggi_Dati_Nel_Tableau(void)
{
	FILE *fdata;
	long i,j,k;
	long no;
	long rr;
	double cc;

	// Apri il file dati
	fdata = fopen("input.dat","r");

	// Leggi il numero di righe e colonne 
	fscanf(fdata,"%d %d",&n,&m);
 
	// Inizializza il Tableau
	for (i=0;i<=m;i++)
		for (j=0;j<=n;j++)
			Mat[i][j]=0.0;

	// Leggi il vettore dei "termini noti"
	for (i=1;i<=m;i++)
		fscanf(fdata,"%lf",&(Mat[i][0]));
 
	// Leggi il vettore dei "versi"
	for (i=1;i<=m;i++)
		fscanf(fdata,"%d",&(Segno[i]));
 
	// Leggi la matrice dei coefficienti
	for (j=1;j<=n;j++)
	{
		// Leggi il costo della colonna j
		fscanf(fdata,"%lf",&cc);
		Mat[0][j]=-cc;
 
		// Leggi il numero di coefficienti non-zero della colonna j
		fscanf(fdata,"%d",&no);

		// Leggi i coefficienti non-zero della colonna j
		for (k=1;k<=no;k++)
		{
			fscanf(fdata,"%d",&rr);
			fscanf(fdata,"%lf",&(Mat[rr][j]));
		}
	}

	// Chiudi il file dati
	fclose(fdata);
}

