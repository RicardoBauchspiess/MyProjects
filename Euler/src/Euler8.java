
public class Euler8 {
	public static void main(String[] args) {
		String sequence = 	"73167176531330624919225119674426574742355349194934"+
							"96983520312774506326239578318016984801869478851843"+
							"85861560789112949495459501737958331952853208805511"+
							"12540698747158523863050715693290963295227443043557"+
							"66896648950445244523161731856403098711121722383113"+
							"62229893423380308135336276614282806444486645238749"+
							"30358907296290491560440772390713810515859307960866"+
							"70172427121883998797908792274921901699720888093776"+
							"65727333001053367881220235421809751254540594752243"+
							"52584907711670556013604839586446706324415722155397"+
							"53697817977846174064955149290862569321978468622482"+
							"83972241375657056057490261407972968652414535100474"+
							"82166370484403199890008895243450658541227588666881"+
							"16427171479924442928230863465674813919123162824586"+
							"17866458359124566529476545682848912883142607690042"+
							"24219022671055626321111109370544217506941658960408"+
							"07198403850962455444362981230987879927244284909188"+
							"84580156166097919133875499200524063689912560717606"+
							"05886116467109405077541002256983155200055935729725"+
							"71636269561882670428252483600823257530420752963450";
		
		int n_digits = sequence.length();
		int cont = 0;
		int cont2 = 0;
		//valores em long para conter o valor máximo 9^13 ~= 2.5x10^12 > 2^32		
		long maior_valor = 0;
		long valor = sequence.charAt(cont)-'0';		
		//primeira sequencia de 13 digitos
		for(cont=2;cont<13;cont++)
		{
			//se encontrar um zero, procura próxima sequencia sem zeros
			if(sequence.charAt(cont) == '0')
			{
				cont++;
				cont2 = 0;
				valor = 1;
				do {
					if(sequence.charAt(cont+cont2)=='0')//se achar um zero reinicia a busca a partir do número seguinte
					{
						cont += cont2+1;
						cont2 = 0;
						valor = 1;
					}
					else
					{
						valor*=sequence.charAt(cont+cont2)-'0';
						cont2++;
					}
				}while((cont2<13)&&(cont+cont2<n_digits));//até achar uma sequencia de 13 números não nulos ou acabarem os números
				cont = cont+12;	
				if(cont2<13)//se terminar a sequencia sem encontrar uma sequencia de 13 números não nulos
				{
					valor = 0;
				}
			}
			else
			{
				valor*=sequence.charAt(cont)-'0';
			}
		}
		maior_valor = valor;
		//Demais sequências
		for(cont = 13;cont<=n_digits;cont++)
		{
			//se encontrar um zero, procura próxima sequencia sem zeros
			if(sequence.charAt(cont) == '0')
			{
				cont++;
				cont2 = 0;
				valor = 1;
				do {
					if(sequence.charAt(cont+cont2)=='0')//se achar um zero reinicia a busca a partir do número seguinte
					{
						cont += cont2+1;
						cont2 = 0;
						valor = 1;
					}
					else
					{
						valor*=sequence.charAt(cont+cont2)-'0';
						cont2++;
					}
				}while((cont2<13)&&(cont+cont2<n_digits));//até achar uma sequencia de 13 números não nulos ou acabarem os números
				cont = cont+12;	
				if(cont2<13)//se terminar a sequencia sem encontrar uma sequencia de 13 números não nulos
				{
					valor = 0;
				}
			}
			else
			{
				valor/=sequence.charAt(cont-13)-'0'; //remove valor mais antigo da sequência de 13 digitos
				valor*=sequence.charAt(cont)-'0'; //multiplica valor anterior pelo novo digito
			}
			if(valor>maior_valor)
			{
				maior_valor = valor;
			}
		}
		System.out.println("Maior multiplicação de 13 dígitos consecutivos:"+maior_valor);	
	}
}
